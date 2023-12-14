import pandas as pd

from src.framework import Component
from src.mining.outliers_detection import train_speed


class TrainTooFastComponent(Component):
    def run(self, **kwargs):
        df = train_speed(kwargs.values().__iter__().__next__())
        df = df[df['speed_too_high']]
        df.sort_index(inplace=True)

        return df
