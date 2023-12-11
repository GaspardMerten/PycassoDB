import pandas as pd
import numpy as np

from src.framework.component import Component
from components.source import SOURCE_DATA_COLS


class ChainSawProcessing(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        # Create a copy of the source dataframe
        df = source.copy()

        # Sort by timestamp
        df.sort_index(inplace=True)

        pre = len(df)

        df["time_difference"] = df.index.to_series().diff()
        # TODO: those are not interesting, we are just removing the first data of each trip
        #       we better use the fact that there is nothing around, not just before the tuple
        # Remove data when consecutive timestamps delta is bigger than 30min
        df = df[df["time_difference"] < pd.Timedelta(minutes=30)]
        # Remove data when consecutive timestamps delta is smaller than 1s
        df = df[df["time_difference"] > pd.Timedelta(seconds=1)]

        post_time = len(df)

        # Note: speed is in m/s
        # Remove data where speed is smaller than 1km/h
        df = df[df['speed'] >= 1 / 3.6]
        # Remove data where speed is bigger than 100km/h
        df = df[df['speed'] <= 100 / 3.6]

        post_speed = len(df)

        print("Source data length:", pre)
        print("Removed", pre - post_time, "rows due to time")
        print("Removed", post_time - post_speed, "rows due to speed")

        # Remove data where the sensors are "not working" (return 0)
        # Is the same as saying, either the sensor is broken, or the system is not running, so we don't need that data
        for col in SOURCE_DATA_COLS:
            df = df[df[col] != 0]

        return df
