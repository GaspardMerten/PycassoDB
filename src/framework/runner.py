import multiprocessing
import time
from datetime import timedelta

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


def run_component(
    storage_manager: StorageManager,
    runner_persistence: RunnerPersistence,
    component: ConfigComponent,
    train_id: str = None,
):
    should_run = True

    data = {}
    max_timestamps = []

    for dependency in component.dependencies:
        last_timestamp = runner_persistence.get_last_timestamp(
            component.name, train_id
        ) + timedelta(seconds=1)

        if not train_id and dependency.get_component.per_train:
            at_least_one_train = False
            for _train_id in storage_manager.retrieve_train_ids():
                if storage_manager.has_sufficient_data_since(
                    last_timestamp,
                    dependency.batch_size,
                    dependency.component,
                    _train_id,
                ):
                    at_least_one_train = True
                    break

            should_run = at_least_one_train
        elif not storage_manager.has_sufficient_data_since(
            last_timestamp,
            dependency.batch_size,
            dependency.component,
            train_id,
        ):
            should_run = False
            break

        if component.per_train and dependency.get_component.per_train:
            data[dependency.component] = storage_manager.get_for_all_trains(
                dependency.component,
                (last_timestamp, None),
                dependency.batch_size,
            )
        elif dependency.get_component.per_train:
            data[dependency.component] = storage_manager.get_for_train(
                dependency.component,
                train_id,
                (last_timestamp, None),
                dependency.batch_size,
            )
        else:
            data[dependency.component] = storage_manager.get_for_agnostic(
                dependency.component,
                (last_timestamp, None),
                dependency.batch_size,
            )

        max_timestamps.append(data[dependency.component].index.max())

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
    runner_persistence = RunnerPersistence(config.runner_persistence, lock=multiprocessing.Lock())

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
