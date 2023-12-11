import tomllib
from dataclasses import dataclass
from typing import List, Dict

__all__ = ["ConfigComponent", "ConfigDependency", "load_config_from_file"]


@dataclass
class ConfigDependency:
    """
    Data class representing a dependency of a configuration component.
    """

    component: str
    batch_size: int = None
    frequency: str = None
    all_trains: bool = False
    components: Dict[str, "ConfigComponent"] = None
    before: bool = False

    @property
    def name(self):
        """
        Property to get the name of the dependent component.
        Ensures that the 'components' attribute is not None before getting the name.
        """
        assert self.components is not None, "Components must not be None"
        return self.components.get(self.component).name + (
            "_before" if self.before else ""
        )

    @property
    def get_component(self):
        """
        Property to get the dependent component.
        Ensures that the 'components' attribute is not None before getting the component.
        """
        assert self.components is not None, "Components must not be None"
        return self.components.get(self.component)


@dataclass
class ConfigComponent:
    """
    Data class for storing configuration of a component.
    """

    name: str
    component_class: str = None
    multiple_outputs: bool = False
    per_train: bool = True
    run_per_train: bool = True
    dependencies: List[ConfigDependency] = None
    settings: dict = None
    outliers_producer: bool = False
    intensity_column: str = None


@dataclass
class Config:
    """
    Data class for the main configuration structure.
    """

    storage_folder: str
    runner_persistence: str
    components: Dict[str, ConfigComponent]


def _parse_components(components) -> Dict[str, ConfigComponent]:
    """
    Parses the components section of the configuration file.

    @param components: The components section of the configuration file.
    @return: A dictionary of parsed components.
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
            outliers_producer=value.pop("outliers_producer", False),
            intensity_column=value.pop("intensity_column", None),
            dependencies=parsed_dependencies,
            # Pop everything else into settings
            settings=value,
        )
        parsed_components[key] = component

    return parsed_components


def _parse_config(data: dict) -> Config:
    """
    Parses the entire configuration file.

    @param data: The parsed configuration file.
    @return: An instance of the Config data class containing the parsed configuration.
    """
    components = _parse_components(data["components"])
    globals_data = data["globals"]

    return Config(
        storage_folder=globals_data.get("storage_folder", "data"),
        runner_persistence=globals_data.get(
            "runner_persistence", "data/persistence.json"
        ),
        components=components,
    )


def load_config_from_file(path: str) -> Config:
    """
    Loads and parses a configuration file.

    @param path: The path to the configuration file.
    @return: An instance of the Config data class containing the parsed configuration.
    """
    with open(path, "rb") as file:
        parsed_data = tomllib.load(file)

    return _parse_config(parsed_data)
