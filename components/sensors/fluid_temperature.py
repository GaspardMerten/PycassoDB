import pandas as pd

from src.framework import Component
from src.mining.outliers_detection import stopped_motors


class FluidOverheatingComponent(Component):
    def run(self, source: pd.DataFrame):
        df = stopped_motors(source)

        df = df[
            df['water_overheating_1'] |
            df['water_overheating_2'] |
            df['oil_overheating_1'] |
            df['oil_overheating_2'] |
            df['air_overheating_1'] |
            df['air_overheating_2']
        ]

        return df
