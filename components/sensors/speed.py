import pandas as pd

from src.framework import Component
from src.mining.outliers_detection import train_speed


class TrainTooFastComponent(Component):
    def run(self, surgery: pd.DataFrame, ):
        print("running train too fast component")
        df = train_speed(surgery)
        df = df[df['speed_too_high']]
        df.sort_index(inplace=True)

        return df
