import pandas as pd
import numpy as np

from src.framework import Component


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


class PreprocessingComponent(Component):
    def run(self, source_before: pd.DataFrame, source: pd.DataFrame):
        """
        This function takes a dataframe as input and cleans it.

        :param source_before:
        :param source: The dataframe to clean
        :return: The cleaned dataframe
        """
        # Sort the dataframe by timestamp (before is reversed)
        source_before.sort_index(inplace=True)

        df_past = source_before.copy()

        if source_before.empty:
            first_batch = True
        else:
            # Find the latest previous data (to create continuity in time difference computation)
            df_past = source_before[source_before.index < source.index[0]]
            df_past = df_past[df_past.index == df_past.index[-1]]

            first_batch = df_past.empty  # To decide whether to remove the date or to change the values to 0

        # Concatenate the two dataframes
        df = pd.concat([df_past, source])
        df.sort_index(inplace=True)  # Ensure that the newly added row is in the first place by sorting

        # Convert all columns to int16
        for col in df.columns:
            if col not in ["timestamp", "lat", "lon"]:
                df[col] = df[col].fillna(0)  # Fill NaN with 0
                df[col] = df[col].astype("int16")

        df['time_difference'] = df.index.to_series().diff()

        # Replace N/A values with 0 seconds
        df['time_difference'] = df['time_difference'].fillna(pd.Timedelta(seconds=0))

        # Compute the distance between each tuple of the given train
        df['distance'] = haversine(
            df['lat'].shift(),
            df['lon'].shift(),
            df['lat'],
            df['lon']
        ) * 1000  # multiplied by 1000 to have it in meters instead of kilometers

        # Compute the speed between each tuple of the given train
        df['speed'] = df['distance'] / df['time_difference'].dt.total_seconds()  # in m/s

        if first_batch:
            # Replace the first distance with 0
            df.loc[df.index[0], 'distance'] = 0
            # Replace the first speed with 0
            df.loc[df.index[0], 'speed'] = 0
        else:
            # Remove the first row
            df = df[1:]

        df.sort_index(inplace=True)

        return df
