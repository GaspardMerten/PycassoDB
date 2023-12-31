import pandas as pd
import numpy as np

from src.framework.component import Component
from components.source import SOURCE_DATA_COLS
from src.mining.outliers_detection import stopped_motors, cooling_sensors


class ChainsawProcessing(Component):
    def run(self, enriched: pd.DataFrame) -> pd.DataFrame:
        # Create a copy of the source dataframe
        df = enriched.copy()

        # Sort by timestamp
        df.sort_index(inplace=True)

        # Remove the N/A values, faulty sensors
        df.dropna()

        # Remove illogical or non important values
        df_labelled = stopped_motors(df)
        df = df[
            # 1. RPM and Oil Pressure
            (df_labelled["faulty_rpm_1"] == False) |
            (df_labelled["faulty_rpm_2"] == False) |
            (df_labelled["faulty_oil_press_1"] == False) |
            (df_labelled["faulty_oil_press_2"] == False) |
            # 2. Train stopped at station
            (df_labelled["stopped_at_station"] == False) |
            # 3. Train stopped at workshop
            (df_labelled["stopped_at_workshop"] == False)
        ]

        # Remove fluid overheating
        # Note: High correlation between Water and Oil temperature
        df_labelled = cooling_sensors(df)
        df = df[
            (df_labelled["water_overheating_1"] ^ df_labelled["oil_overheating_1"]) |
            (df_labelled["water_overheating_2"] ^ df_labelled["oil_overheating_2"])
        ]

        # Compute time between each tuple received
        df["time_difference"] = df.index.to_series().diff()

        # Remove the row that corresponds to data sent directly after another one (less than 10 seconds)
        # Tend to create glitch in measurements, typically GPS measurements moving fast because resampling with error
        df = df[df["time_difference"] > pd.Timedelta(seconds=10)]

        # Recompute the time difference between each tuple
        df["time_difference"] = df.index.to_series().diff()

        # Remove isolated data (10 minutes before and after)
        threshold = pd.Timedelta(minutes=10)

        df["time_difference_before"] = df.index.to_series().diff(periods=-1)
        df["isolated"] = (df["time_difference_before"] < -threshold) & (df["time_difference"] > threshold)

        # Note: speed is in m/s
        # Remove data where speed is smaller than 1km/h but without too much time since last tuple
        df = df[(df["speed"] >= 1 / 3.6) & (df["time_difference"] < threshold)]
        # Remove data where speed is bigger than 100km/h
        df = df[df["speed"] <= 120 / 3.6]

        # Remove data where the sensors are "not working" (return 0)
        # Is the same as saying, either the sensor is broken, or the system is not running, so we don't need that data
        for col in SOURCE_DATA_COLS:
            df = df[df[col] != 0]

        return df
