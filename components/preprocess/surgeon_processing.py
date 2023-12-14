from uuid import uuid4

import pandas as pd

from src.framework.component import Component
from src.mining import find_trip_split_indexes


class SurgeonProcessing(Component):
    def run(self, enriched_before: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
        first_batch = False
        if enriched_before.index.min() == enriched.index.min():
            first_batch = True

        # Concatenate the two dataframes
        df = pd.concat([enriched_before, enriched])

        # Divide the data into segments (threshold: 10 minutes)
        segments = find_trip_split_indexes(df, 10)

        # For each segment, decide on a UID
        for i, segment in enumerate(segments):
            # Generate a random ID
            uuid = str(uuid4())

            # Set the UID for the segment
            df.loc[segment[0] : segment[1], "uuid"] = uuid

        # Remove the rows that are before enriched.index.min()
        # which do not have the same UID as the first row of enriched
        if not first_batch:  # But, only if it is not the first batch
            df_past = df[df.index < enriched.index.min()]

            # Find the UID of the first row of enriched
            uuid = df.loc[enriched.index.min(), "uuid"]

            # Remove the rows that do not have the same UID
            df_past = df_past[df_past["uuid"] != uuid]

            # Remove the rows
            df = df[~df.index.isin(df_past.index)]

        df.sort_index(inplace=True)

        return df
