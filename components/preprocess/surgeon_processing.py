import uuid

import pandas as pd

from src.framework.component import Component
from src.mining import find_trip_split_indexes


class SurgeonProcessing(Component):
    def run(self, source_before: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
        first_batch = False
        if source_before.index.min() == source.index.min():
            first_batch = True

        print("First batch:", first_batch)

        print("Source before:", source_before.index.min(), "-", source_before.index.max())
        print("Size:", source_before.shape[0])

        print("Source:", source.index.min(), "-", source.index.max())
        print("Size:", source.shape[0])

        # Concatenate the two dataframes
        df = pd.concat([source_before, source])

        print("Concatenated:", df.index.min(), "-", df.index.max())
        print("Size:", df.shape[0])

        # Remove Duplicates
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)

        print("Size after duplicates removed:", df.shape[0])

        # Divide the data into segments (threshold: 10 minutes)
        segments = find_trip_split_indexes(df, 10)

        print("Segments:", len(segments))
        print(segments)

        # For each segment, decide on a UID
        for i, segment in enumerate(segments):
            # Generate a random ID
            uid = str(uuid.uuid4())

            # Set the UID for the segment
            df.loc[segment[0]:segment[1], 'uid'] = uid

        # Remove the rows that are before source.index.min()
        # which do not have the same UID as the first row of source
        if not first_batch:  # But, only if it is not the first batch
            df_past = df[df.index < source.index.min()]

            # Find the UID of the first row of source
            uid = df.loc[source.index.min(), 'uid']

            # Remove the rows that do not have the same UID
            df_past = df_past[df_past['uid'] != uid]

            # Remove the rows
            df = df[~df.index.isin(df_past.index)]

        df.sort_index(inplace=True)

        print("Size after removing past:", df.shape[0])
        print(df.head())

        return df
