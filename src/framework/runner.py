import logging
import multiprocessing
import time
from typing import Union

import pandas as pd

from src.framework.config import ConfigComponent, load_config_from_file
from src.framework.period import build_period_from_frequency
from src.framework.runner_persistence import RunnerPersistence
from src.framework.storage import StorageManager

__all__ = ["run_pipeline"]


def _instantiate_component_from_config(component: ConfigComponent):
    """Instantiate a component from a component class string."""
    file = ".".join(component.component_class.split(".")[:-1])
    class_name = component.component_class.split(".")[-1]
    module = __import__(file, fromlist=[class_name])
    class_ = getattr(module, class_name)
    # Instantiate class
    instance = class_(component.settings)

    return instance


def _component_should_run(
    dependency, get_name, is_before, period, storage_manager, train_id
):
    """
    Check if the component should run, depending on the dependency. If the dependency is before, then the component
    skips the dependency (since past data is not required by definition). Otherwise, the component checks if the
    dependency has sufficient data. If the dependency has sufficient data, then the component should run. Otherwise,
    the component should not run.
    """
    if is_before:
        return True

    start_timestamp = period[1] if dependency.frequency else period[0]

    if not train_id and dependency.get_component.per_train:
        for _train_id in storage_manager.retrieve_train_ids():
            if storage_manager.has_sufficient_data_since(
                start_timestamp,
                dependency.batch_size,
                get_name,
                _train_id,
            ):
                return True
        return False
    elif not storage_manager.has_sufficient_data_since(
        start_timestamp,
        dependency.batch_size,
        get_name,
        train_id,
    ):
        return False

    return True


def _run_component_once(
    storage_manager: StorageManager,
    runner_persistence: RunnerPersistence,
    component: ConfigComponent,
    train_id: str = None,
):
    """
    Run a component once. The caller is responsible for specifying whether the component should run for all trains or
    for a specific train (this behaviour is defined in the component configuration).

    If the component never ran the method get_starting_timestamp_for_all_dependencies will be called to get the minimum
    timestamp for all dependencies. Otherwise, the last timestamp will be retrieved from the runner persistence.

    The last timestamp is defined as the maximum timestamp of all used dependencies. If a period is used, then the
    maximum timestamp of the period is used instead of the last timestamp of the values.

    The component will run if at least one dependency is not before. Otherwise, the system will raise an assertion error.

    :param storage_manager:
    :param runner_persistence:
    :param component:
    :param train_id:
    :return:
    """
    should_run = True

    data = {}
    max_timestamps = []

    # Check if at least one dependency is not before (if no dependency, then it is not before)
    has_one_not_before = len(component.dependencies) == 0

    starting_timestamp = _get_starting_timestamp_for_all_dependencies(
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

        if is_before:
            period = (pd.Timestamp.min, last_timestamp)
        else:
            period = (last_timestamp + pd.Timedelta(seconds=1), pd.Timestamp.now())

        if dependency.batch_size:
            limit = dependency.batch_size

        if dependency.frequency:
            period = build_period_from_frequency(
                dependency.frequency, last_timestamp, is_before
            )

        # Different from the dependency name, which is the name of the component + _before if before is True,
        # the component name is the name of the component without the _before suffix, used to retrieve the data,
        # while the dependency name is used for running the component.
        component_name = dependency.component

        # Check if the component should run, depending on the dependency
        should_run = should_run and _component_should_run(
            dependency,
            component_name,
            is_before,
            period,
            storage_manager,
            train_id,
        )

        data[dependency.name] = _get_data_from_dependency(
            component,
            component_name,
            dependency,
            is_before,
            limit,
            period,
            storage_manager,
            train_id,
        )

        if not dependency.before:
            if dependency.frequency:
                max_timestamps.append(period[1])
            else:
                max_timestamps.append(data[component_name].index.max())

    assert (
        has_one_not_before
    ), "At least one dependency should not be before, otherwise the component will always run on the first data"

    if should_run:
        logging.info(f"Running component {component.name} for train {train_id}")
        # Instantiate component
        instance = _instantiate_component_from_config(component)
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


def _get_data_from_dependency(
    component,
    component_name,
    dependency,
    is_before,
    limit,
    period,
    storage_manager,
    train_id,
):
    if component.run_per_train and dependency.get_component.per_train:
        return storage_manager.get_for_train(
            component_name,
            train_id,
            period,
            limit,
            invert=is_before,
        )
    elif dependency.get_component.per_train:
        return storage_manager.get_for_all_trains(
            component_name,
            period,
            limit,
            invert=is_before,
        )
    else:
        return storage_manager.get_for_agnostic(
            component_name,
            period,
            limit,
            invert=is_before,
        )


def _get_starting_timestamp_for_all_dependencies(
    component: ConfigComponent,
    runner_persistence: RunnerPersistence,
    storage_manager: StorageManager,
    train_id: str = None,
) -> Union[pd.Timestamp, None]:
    """
    If the component never ran, then get the minimum timestamp for all dependencies. Otherwise, return None.
    """
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


def _run_component_for_ever(
    storage_manager: StorageManager,
    runner_persistence: RunnerPersistence,
    component: ConfigComponent,
):
    """
    Run a component forever. If the component is run per train, then it will run for each train.
    """
    while True:
        if component.run_per_train:
            ids = storage_manager.retrieve_train_ids()
            for train_id in ids:
                _run_component_once(
                    storage_manager, runner_persistence, component, train_id
                )
        else:
            _run_component_once(storage_manager, runner_persistence, component)

        time.sleep(.1)


def run_pipeline(config_path: str = "config.toml"):
    """
    Run all components in the configuration file in separate processes. (Also instantiates the storage manager and the
    runner persistence.)

    :param config_path: The path to the configuration file (default: config.toml)
    :return: None
    """

    # Load the global and component configuration
    config = load_config_from_file(config_path)
    # Instantiate the storage manager, which handles all data storage for the components
    storage_manager = StorageManager(config.storage_folder)
    # Instantiate the runner persistence, which handles the persistence of the runner
    runner_persistence = RunnerPersistence(
        config.runner_persistence, lock=multiprocessing.Lock()
    )

    # Run all components in separate processes
    processes = []

    for component in config.components.values():
        process = multiprocessing.Process(
            target=_run_component_for_ever,
            args=(storage_manager, runner_persistence, component),
        )
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
