import pandas as pd

from src.framework import Component


class NotApplicableCountComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """
        This function return the rows where the value is not applicable.

        :param source: The dataframe containing the points
        :return: The rows where the value is not applicable
        """
        # Get the rows where any value is not applicable
        return source[source.isna().any(axis=1)]
