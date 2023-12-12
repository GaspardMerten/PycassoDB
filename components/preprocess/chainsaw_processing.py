import pandas as pd
import numpy as np

from src.framework.component import Component
from components.source import SOURCE_DATA_COLS


class ChainsawProcessing(Component):
    def run(self, enriched: pd.DataFrame) -> pd.DataFrame:
        # Create a copy of the source dataframe
        df = enriched.copy()

        # Sort by timestamp
        df.sort_index(inplace=True)

        pre = len(df)

        df["time_difference"] = df.index.to_series().diff()

        # Remove the row that corresponds to data sent directly after another one (less than 10 seconds)
        df = df[df["time_difference"] > pd.Timedelta(seconds=10)]

        df["time_difference"] = df.index.to_series().diff()

        # Remove isolated data (10 minutes before and after)
        df["time_difference_before"] = df.index.to_series().diff(periods=-1)

        threshold = pd.Timedelta(minutes=10)
        df["isolated"] = df["time_difference_before"] < -threshold & df["time_difference"] > threshold

        post_time = len(df)

        # Note: speed is in m/s
        # Remove data where speed is smaller than 1km/h
        df = df[df['speed'] >= 1 / 3.6]
        # Remove data where speed is bigger than 100km/h
        df = df[df['speed'] <= 120 / 3.6]

        post_speed = len(df)

        print("Source data length:", pre)
        print("Removed", pre - post_time, "rows due to time")
        print("Removed", post_time - post_speed, "rows due to speed")

        # Remove data where the sensors are "not working" (return 0)
        # Is the same as saying, either the sensor is broken, or the system is not running, so we don't need that data
        for col in SOURCE_DATA_COLS:
            df = df[df[col] != 0]

        return df
