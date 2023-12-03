import time
import tomllib
from dataclasses import dataclass
from datetime import timedelta
from typing import List

from src.framework.storage import StorageManager


@dataclass
class Dependency:
    component: str
    batch_size: int
    all_trains: bool = False


@dataclass
class Component:
    name: str
    component_class: str = None
    multiple_outputs: bool = False
    per_train: bool = True
    dependencies: List[Dependency] = None
    settings: dict = None


# Parsing the TOML content
parsed_data = tomllib.load(open("config.toml", "rb"))


# Extracting components and converting them into dataclasses
def parse_components(data):
    components = data.get("components", {})
    parsed_components = {}
    for key, value in components.items():
        dependencies = value.pop("dependencies", [])
        parsed_dependencies = [Dependency(**dep) for dep in dependencies]

        component = Component(
            name=key,
            component_class=value.pop("class"),
            multiple_outputs=value.pop("multiple_outputs", False),
            per_train=value.pop("per_train", True),
            dependencies=parsed_dependencies,
            settings=value,
        )
        parsed_components[key] = component

    return parsed_components


components = parse_components(parsed_data)
storage_manager = StorageManager("data_2")


def instantiate_component(component: Component):
    """Instantiate a component from a component class string."""
    file = ".".join(component.component_class.split(".")[:-1])
    class_name = component.component_class.split(".")[-1]
    module = __import__(file, fromlist=[class_name])
    class_ = getattr(module, class_name)
    # Instantiate class
    instance = class_(component.settings)

    return instance


def run_component(component: Component, train_id: str = None):
    print("Running component", component.name, component.dependencies)
    should_run = True

    last_timestamp = storage_manager.get_last_timestamp(
        component.name, train_id
    ) + timedelta(seconds=1)

    data = {}

    for dependency in component.dependencies:
        print("Checking dependency", dependency.component)
        if not storage_manager.has_sufficient_data_since(
                last_timestamp,
                dependency.batch_size,
                dependency.component,
                train_id,
        ):
            should_run = False
            break

        if (
                components[dependency.component].per_train
                and dependency.all_trains
        ):
            data[dependency.component] = storage_manager.get_for_all_trains(
                dependency.component,
                (last_timestamp, None),
                dependency.batch_size,
            )
        elif components[dependency.component].per_train:
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

    if should_run:
        # Instantiate component
        instance = instantiate_component(component)
        # Run component
        if component.multiple_outputs:
            dfs = instance.run(**data)
            for df in dfs:
                storage_manager.store(df, component.name)
        else:
            df = instance.run(**data)
            storage_manager.store(df, component.name)


while True:
    # Run all components
    for component in components.values():
        print("Checking component", component.name)
        if component.per_train:
            print("X")
            for train_id in storage_manager.retrieve_train_ids():
                run_component(component, train_id)
        else:
            print("Y")
            run_component(component)

    time.sleep(1)