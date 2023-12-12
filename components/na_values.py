import pandas as pd

from src.framework import Component


class NotApplicableCountComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """
        This function return the rows where the value is not applicable.

        :param source: The dataframe containing the points
        :return: The rows where the value is not applicable
        """
        df = source.copy()

        # Get the rows where the value is not applicable
        df = df[df.isna().any(axis=1)]

        print(len(df))

        return df
