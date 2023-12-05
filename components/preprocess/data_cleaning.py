import os

import pandas as pd
import numpy as np

from src.framework.component import Component
from components.source import SOURCE_DATA_COLS


# vectorized haversine function
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.
    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))  # return the value in km since the earth_radius is in km


class DataCleaning(Component):
    def run(self) -> None:
        source_file = self.config.get("source")

        if not os.path.exists(self.config.get("source")):
            print(f"Source file {source_file} does not exist")
            return None

        df = pd.read_csv(source_file, sep=";")
        df['timestamps_UTC'] = pd.to_datetime(df['timestamps_UTC'])

        df = df.dropna()

        # Compute the time interval between each tuple of a given train
        df = df.sort_values(by=['mapped_veh_id', 'timestamps_UTC'])
        df['time_difference'] = df.groupby(['mapped_veh_id'])['timestamps_UTC'].diff()

        # Replace N/A values with 0 seconds
        df['time_difference'] = df['time_difference'].fillna(pd.Timedelta(seconds=0))

        # Compute the relative distance and average speed between each tuple for each train separately
        for vehicle in df['mapped_veh_id'].unique():
            # Get the index of the tuples of the given train
            vehicle_idx = df[df['mapped_veh_id'] == vehicle].index

            # Compute the distance between each tuple of the given train
            df.loc[vehicle_idx, 'distance'] = haversine(
                df.loc[vehicle_idx, 'lat'].shift(),
                df.loc[vehicle_idx, 'lon'].shift(),
                df.loc[vehicle_idx, 'lat'],
                df.loc[vehicle_idx, 'lon']
            ) * 1000  # multiplied by 1000 to have it in meters instead of kilometers
            # Replace the first distance with 0
            df.loc[vehicle_idx[0], 'distance'] = 0

            # Compute the speed between each tuple of the given train
            df.loc[vehicle_idx, 'speed'] = df.loc[vehicle_idx, 'distance'] / df.loc[
                vehicle_idx, 'time_difference'
            ].dt.total_seconds()  # in m/s
            # Replace the first speed with 0
            df.loc[vehicle_idx[0], 'speed'] = 0

        # Drop the index
        df = df.reset_index(drop=True)

        # Store the processed dataframe
        df.to_csv(self.config.get("output"), sep=";", index=False)
