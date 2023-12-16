import abc

import pandas as pd

__all__ = ["Component"]


class Component(abc.ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.debug = self.config.get("debug", False)

    @abc.abstractmethod
    def run(self, **sources) -> pd.DataFrame:
        """Run the component."""
        pass
