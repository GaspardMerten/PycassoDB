import pandas as pd

from src.framework import Component
from src.mining.outliers_detection import stopped_motors


class FluidOverheatingComponent(Component):
    def run(self, source: pd.DataFrame):
        df = source.copy(tt)

        df = df

        return df
