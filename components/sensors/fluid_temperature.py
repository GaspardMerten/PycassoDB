import pandas as pd

from src.framework import Component
from src.mining.outliers_detection import stopped_motors, cooling_sensors


class FluidOverheatingComponent(Component):
    def run(self, **kwargs):
        df = cooling_sensors(kwargs.values().__iter__().__next__())
        df = df[
            df["water_overheating_1"]
            | df["water_overheating_2"]
            | df["oil_overheating_1"]
            | df["oil_overheating_2"]
            | df["air_overheating_1"]
            | df["air_overheating_2"]
        ]

        return df
