import pandas as pd

from src.framework import Component


class NotApplicableCountComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """
        This function return the rows where the value is not applicable.

        :param source: The dataframe containing the points
        :return: The rows where the value is not applicable
        """

        # Keep only rows where at least one value is not applicable/null
        df = source[source.isnull().any() | source.isna().any() | (source == "na").any()]
        df.sort_index(inplace=True)
        return df
