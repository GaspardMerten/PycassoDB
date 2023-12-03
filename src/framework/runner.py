import tomllib
from dataclasses import dataclass
from typing import List


@dataclass
class Dependency:
    component: str
    batch_size: int


@dataclass
class Component:
    file: str = None
    multiple_outputs: bool = None
    dependencies: List[Dependency] = None
    window: int = None
    column: str = None
    sensitivity: int = None


# Parsing the TOML content
parsed_data = tomllib.load("config.toml")


# Extracting components and converting them into dataclasses
def parse_components(data):
    components = data.get("components", {})
    parsed_components = {}
    for key, value in components.items():
        dependencies = value.get("dependencies", [])
        parsed_dependencies = [Dependency(**dep) for dep in dependencies]
        component = Component(
            multiple_outputs=value.get("multiple_outputs"),
            dependencies=parsed_dependencies if dependencies else None,
            window=value.get("window"),
            column=value.get("column"),
            sensitivity=value.get("sensitivity"),
        )
        parsed_components[key] = component
    return Components(**parsed_components)
