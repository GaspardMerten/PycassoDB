import pandas as pd

from src.framework import Component
from src.mining.outliers_detection import train_speed


class TrainTooFastComponent(Component):
    def run(self, source: pd.DataFrame):
        df = train_speed(source)

        df = df[df['speed_too_high']]

        return df
