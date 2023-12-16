from uuid import uuid4

import pandas as pd

from src.framework.component import Component
from src.mining import find_trip_split_indexes


class SurgeonProcessing(Component):
    def run(self, enriched: pd.DataFrame) -> pd.DataFrame:
        # Divide the data into segments (threshold: 10 minutes)
        segments = find_trip_split_indexes(enriched, 10)
        df = enriched.copy()

        # Segments is a list of slices of the dataframe, insert random uuids to identify each segment
        for i, segment in enumerate(segments):
            df.loc[segment[0] : segment[1], "uuid"] = str(uuid4())

        # Drop where segment_id is null
        df.dropna(subset=["uuid"], inplace=True)

        return df
