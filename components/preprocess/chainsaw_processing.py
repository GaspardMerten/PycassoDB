import pandas as pd
import numpy as np

from src.framework.component import Component
from components.source import SOURCE_DATA_COLS


class ChainSawProcessing(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        # Create a copy of the source dataframe
        df = source.copy()

        # Sort by timestamps_UTC
        timestamp_col = 'timestamps_UTC'
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(by=timestamp_col)
        # Remove data when consecutive timestamps delta is bigger than 30min
        df = df[df[timestamp_col].diff() < pd.Timedelta(minutes=30)]
        # Remove data when consecutive timestamps delta is smaller than 1s
        df = df[df[timestamp_col].diff() > pd.Timedelta(seconds=1)]

        # Note: speed is in m/s
        # Remove data where speed is smaller than 1km/h
        df = df[df['speed'] >= 1 / 3.6]
        # Remove data where speed is bigger than 100km/h
        df = df[df['speed'] <= 100 / 3.6]

        # Remove data where the sensors are "not working" (return 0)
        for col in SOURCE_DATA_COLS:
            df = df[df[col] != 0]

        # Drop the index
        df = df.reset_index(drop=True)

        return df
