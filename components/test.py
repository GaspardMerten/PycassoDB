import pandas as pd

from src.framework.component import Component


class TestOutliers(Component):
    def run(self, source_before: pd.DataFrame, source: pd.DataFrame):
        # Print min and max timestamps for each dataframe
        print("Source before")
        print(source_before.index.min())
        print(source_before.index.max())

        print("Source")
        print(source.index.min())
        print(source.index.max())

        # Print count of rows for each dataframe
        print("Source before")

        print(source_before.shape[0])
        print("Source")

        print(source.shape[0])

        # return a df only with first row of source and source_before
        return pd.concat([source_before.head(1), source.head(1)])



