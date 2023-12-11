import multiprocessing
import time
from datetime import timedelta, datetime

import pandas as pd

from src.framework.config import ConfigComponent, load_config
from src.framework.runner_persistence import RunnerPersistence
from src.framework.storage import StorageManager


def instantiate_component(component: ConfigComponent):
    """Instantiate a component from a component class string."""
    file = ".".join(component.component_class.split(".")[:-1])
    class_name = component.component_class.split(".")[-1]
    module = __import__(file, fromlist=[class_name])
    class_ = getattr(module, class_name)
    # Instantiate class
    instance = class_(component.settings)

    return instance


def _build_period_from_frequency(
    frequency: str, last_timestamp: pd.Timestamp, before: bool
):
    """
    Build a period from a frequency and a last timestamp. A frequency is [0-9]+[h|d|w|m]. The period is the next/previous
    one from the last timestamp, rounded to the period.
    :param frequency: The frequency
    :param last_timestamp: The last timestamp
    :param before: Whether to get the previous period or the next one
    :return: A period (start, end)
    """

    # Get the period from the frequency
    period = frequency[-1]
    period = {
        "h": "H",
        "d": "D",
        "w": "W",
        "m": "M",
    }[period]

    timedelta_period = {
        "H": timedelta(hours=1),
        "D": timedelta(days=1),
        "W": timedelta(weeks=1),
        "M": timedelta(days=30),
    }[period]

    # Get the number of periods from the frequency
    n_periods = int(frequency[:-1])

    # Get the last timestamp rounded to the period
    last_timestamp = last_timestamp.round(period)

    # Convert last timestamp to a datetime
    last_timestamp = last_timestamp.to_pydatetime()

    # Get the next/previous period
    if before:
        try:
            start = last_timestamp - timedelta_period * n_periods
        except OverflowError:
            start = datetime.min
        end = last_timestamp
    else:
        start = last_timestamp
        end = last_timestamp + timedelta_period * n_periods
    # Reconverting to pandas timestamp

    return start, end


def run_component(
    storage_manager: StorageManager,
    runner_persistence: RunnerPersistence,
    component: ConfigComponent,
    train_id: str = None,
):
    should_run = True

    data = {}
    max_timestamps = []

    # Check if at least one dependency is not before (if no dependency, then it is not before)
    has_one_not_before = len(component.dependencies) == 0

    starting_timestamp = get_starting_timestamp_for_all_dependecies(
        component, runner_persistence, storage_manager, train_id
    )

    # For each dependency, check if the component should run
    for dependency in component.dependencies:
        has_one_not_before = has_one_not_before or not dependency.before

        # Get last timestamp for the dependency
        last_timestamp = starting_timestamp or runner_persistence.get_last_timestamp(
            component.name, train_id
        )

        is_before = dependency.before

        limit = None
        period = (last_timestamp + pd.Timedelta(seconds=1), pd.Timestamp.now())

        if dependency.batch_size:
            limit = dependency.batch_size

        if dependency.frequency:
            period = _build_period_from_frequency(
                dependency.frequency, last_timestamp, is_before
            )

        get_name = dependency.component
        get_complete_name = get_name + ("_before" if is_before else "")
        if not is_before:
            if not train_id and dependency.get_component.per_train:
                at_least_one_train = False
                for _train_id in storage_manager.retrieve_train_ids():
                    if storage_manager.has_sufficient_data_since(
                        period[1] if dependency.frequency else period[0],
                        dependency.batch_size,
                        get_name,
                        _train_id,
                    ):
                        at_least_one_train = True
                        break
                should_run = at_least_one_train
            elif not storage_manager.has_sufficient_data_since(
                period[1] if dependency.frequency else period[0],
                dependency.batch_size,
                get_name,
                train_id,
            ):
                should_run = False
                break
        if component.run_per_train and dependency.get_component.per_train:
            data[get_complete_name] = storage_manager.get_for_train(
                get_name,
                train_id,
                period,
                limit,
                invert=is_before,
            )
        elif dependency.get_component.per_train:
            data[get_complete_name] = storage_manager.get_for_all_trains(
                get_name,
                period,
                limit,
                invert=is_before,
            )
        else:
            data[get_complete_name] = storage_manager.get_for_agnostic(
                get_name,
                period,
                limit,
                invert=is_before,
            )

        if not dependency.before:
            if dependency.frequency:
                max_timestamps.append(period[1])
            else:
                max_timestamps.append(data[get_name].index.max())

    assert (
        has_one_not_before
    ), "At least one dependency should not be before, otherwise the component will always run on the first data"

    if should_run:
        print("Running component", component.name)
        # Instantiate component
        instance = instantiate_component(component)
        # Run component
        if component.multiple_outputs:
            dfs = instance.run(**data)
            for df in dfs:
                storage_manager.store(df, component.name, train_id)
        else:
            df = instance.run(**data)
            storage_manager.store(df, component.name, train_id)

        if max_timestamps:
            runner_persistence.register_last_timestamp(
                component.name, min(max_timestamps), train_id
            )


def get_starting_timestamp_for_all_dependecies(
    component, runner_persistence, storage_manager, train_id
):
    min_timestamp = None

    train_ids = storage_manager.retrieve_train_ids()

    # Get min_timestamp for all dependencies if component never ran
    if not runner_persistence.get_did_run(component.name, train_id):
        for dependency in component.dependencies:
            first_timestamp = storage_manager.get_first_timestamp(
                dependency.component,
                train_ids[0]
                if train_ids and dependency.get_component.per_train
                else None,
            )

            if first_timestamp:
                if not min_timestamp:
                    min_timestamp = first_timestamp
                else:
                    min_timestamp = min(min_timestamp, first_timestamp)
    return min_timestamp


def run_component_process(storage_manager, runner_persistence, component):
    while True:
        if component.run_per_train:
            ids = storage_manager.retrieve_train_ids()
            for train_id in ids:
                run_component(storage_manager, runner_persistence, component, train_id)
        else:
            run_component(storage_manager, runner_persistence, component)

        time.sleep(5)


def run(config_path="config.toml"):
    config = load_config(config_path)
    storage_manager = StorageManager(config.storage_folder)
    runner_persistence = RunnerPersistence(
        config.runner_persistence, lock=multiprocessing.Lock()
    )

    processes = []

    for component in config.components.values():
        # Create a separate process for each component
        process = multiprocessing.Process(
            target=run_component_process,
            args=(storage_manager, runner_persistence, component),
        )
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
