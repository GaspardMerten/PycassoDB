import os
import shutil
import tempfile

import pandas as pd

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
                if col not in ["timestamp"]:
                    # Fill NaN with 0
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].astype("int16")

            df.set_index("timestamp", inplace=True)
            df = df.sort_index()

            yield df

        # Remove tmp directory
        shutil.rmtree(tempdir)

        # Remove source file
        os.remove(self.config.get("source"))
