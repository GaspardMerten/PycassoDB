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


def preload_data():
    try:
        processed_df = pd.read_csv('../../assets/processed.csv', sep=";")
        processed_df['timestamps_UTC'] = pd.to_datetime(processed_df['timestamps_UTC'])

        return processed_df
    except FileNotFoundError:
        return None


class DataCleaning(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        # TODO: should add a cache bypass option
        if df := preload_data() is not None:
            print("Processed data found, using it...")
            return df

        print("Processed data not found, creating it...")
        # Create a copy of the source dataframe
        processed_df = source.copy()

        processed_df = processed_df.dropna()

        # Compute the time interval between each tuple of a given train
        processed_df = processed_df.sort_values(by=['mapped_veh_id', 'timestamps_UTC'])
        processed_df['time_difference'] = processed_df.groupby(['mapped_veh_id'])['timestamps_UTC'].diff()

        # Replace N/A values with 0 seconds
        processed_df['time_difference'] = processed_df['time_difference'].fillna(pd.Timedelta(seconds=0))

        # Compute the relative distance and average speed between each tuple for each train separately
        for vehicle in processed_df['mapped_veh_id'].unique():
            # Get the index of the tuples of the given train
            vehicle_idx = processed_df[processed_df['mapped_veh_id'] == vehicle].index

            # Compute the distance between each tuple of the given train
            processed_df.loc[vehicle_idx, 'distance'] = haversine(
                processed_df.loc[vehicle_idx, 'lat'].shift(),
                processed_df.loc[vehicle_idx, 'lon'].shift(),
                processed_df.loc[vehicle_idx, 'lat'],
                processed_df.loc[vehicle_idx, 'lon']
            ) * 1000  # multiplied by 1000 to have it in meters instead of kilometers
            # Replace the first distance with 0
            processed_df.loc[vehicle_idx[0], 'distance'] = 0

            # Compute the speed between each tuple of the given train
            processed_df.loc[vehicle_idx, 'speed'] = processed_df.loc[vehicle_idx, 'distance'] / processed_df.loc[
                vehicle_idx, 'time_difference'
            ].dt.total_seconds()  # in m/s
            # Replace the first speed with 0
            processed_df.loc[vehicle_idx[0], 'speed'] = 0

        # Drop the index
        processed_df = processed_df.reset_index(drop=True)

        # Store the processed dataframe
        processed_df.to_csv('../../assets/processed.csv', sep=";", index=False)

        return processed_df
