import os
import shutil
import tempfile

import pandas as pd
import numpy as np

from src.framework.component import Component


MATCHING = {
    "RS_E_InAirTemp_PC1": "air_temp_1",
    "RS_E_InAirTemp_PC2": "air_temp_2",
    "RS_E_OilPress_PC1": "oil_press_1",
    "RS_E_OilPress_PC2": "oil_press_2",
    "RS_E_RPM_PC1": "rpm_1",
    "RS_E_RPM_PC2": "rpm_2",
    "RS_E_WatTemp_PC1": "water_temp_1",
    "RS_E_WatTemp_PC2": "water_temp_2",
    "RS_T_OilTemp_PC1": "oil_temp_1",
    "RS_T_OilTemp_PC2": "oil_temp_2",
}

SOURCE_DATA_COLS = list(MATCHING.values())


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


class SourceComponent(Component):
    def run(self) -> None:
        print("Running component", self.config.get("source"))
        if not os.path.exists(self.config.get("source")):
            return None

        data = pd.read_csv(self.config.get("source"), sep=";")

        # Create tmp directory
        tempdir = tempfile.mkdtemp()

        # Split into files for each train_id
        for train_id, df in data.groupby("mapped_veh_id"):
            # Sort by timestamps_UTC
            df.to_csv(f"{tempdir}/{train_id}.csv", sep=";", index=False)

        for train_id in os.listdir(tempdir):
            df = pd.read_csv(f"{tempdir}/{train_id}", sep=";")

            df["timestamp"] = pd.to_datetime(df["timestamps_UTC"])

            #  Drop timestamps_UTC column
            df = df.drop(columns=["timestamps_UTC"])

            vehicle_id_col = "mapped_veh_id"

            df = df.rename(columns={vehicle_id_col: "train_id", **MATCHING})

            # Convert all columns to int16
            for col in df.columns:
                if col not in ["timestamp", "lat", "lon"]:
                    # Fill NaN with 0
                    df[col] = df[col].fillna(0)  # TODO: May want to note the number of N/A values of each train
                    df[col] = df[col].astype("int16")

            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

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
            # Replace the first distance with 0
            df.loc[df.index[0], 'distance'] = 0

            # Compute the speed between each tuple of the given train
            df['speed'] = df['distance'] / df['time_difference'].dt.total_seconds()  # in m/s
            # Replace the first speed with 0
            df.loc[df.index[0], 'speed'] = 0

            df.sort_index(inplace=True)

            print(df[["lon", "lat", "distance", "time_difference", "speed"]].head())

            yield df

        # Remove tmp directory
        shutil.rmtree(tempdir)

        # Remove source file
        os.remove(self.config.get("source"))
