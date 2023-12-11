import tomllib
from dataclasses import dataclass
from typing import List, Dict

__all__ = ["ConfigComponent", "ConfigDependency", "load_config"]


@dataclass
class ConfigDependency:
    component: str
    batch_size: int = None
    frequency: str = None
    all_trains: bool = False
    components: Dict[str, "ConfigComponent"] = None
    before: bool = False

    @property
    def get_component(self):
        assert self.components is not None
        return self.components.get(self.component)


@dataclass
class ConfigComponent:
    name: str
    component_class: str = None
    multiple_outputs: bool = False
    per_train: bool = True
    run_per_train: bool = True
    dependencies: List[ConfigDependency] = None
    settings: dict = None


@dataclass
class Config:
    storage_folder: str
    runner_persistence: str
    components: Dict[str, ConfigComponent]


def _parse_components(components) -> Dict[str, ConfigComponent]:
    """
    Extracting components and converting them into dataclasses
    :param components: The source config data as multi-level dictionary
    :return: A dictionary of components (name -> ConfigComponent)
    """
    parsed_components = {}
    for key, value in components.items():
        dependencies = value.pop("dependencies", [])
        parsed_dependencies = [
            ConfigDependency(**dep, components=parsed_components)
            for dep in dependencies
        ]

        component = ConfigComponent(
            name=key,
            component_class=value.pop("class"),
            multiple_outputs=value.pop("multiple_outputs", False),
            per_train=value.pop("per_train", True),
            run_per_train=value.pop("run_per_train", True),
            dependencies=parsed_dependencies,
            settings=value,
        )
        parsed_components[key] = component

    return parsed_components


def _parse_config(data) -> Config:
    components = _parse_components(data["components"])
    globals_data = data["globals"]

    return Config(
        storage_folder=globals_data.get("storage_folder", "data"),
        runner_persistence=globals_data.get(
            "runner_persistence", "data/persistence.json"
        ),
        components=components,
    )


def load_config(path: str) -> Config:
    parsed_data = tomllib.load(open(path, "rb"))

    return _parse_config(parsed_data)
