import pandas as pd

from src.framework.component import Component
from src.mining.moving_average import moving_average
from src.mining.timeseries_outliers import compute_outliers


class MovingAverageOutlierDetector(Component):
    def run(self, source_df: pd.DataFrame) -> pd.DataFrame:
        """Run the component."""
        column = self.config.get("column")
        window = self.config.get("window")

        ts = pd.Series(source_df[column], index=source_df.index)

        # Compute moving average
        ma = moving_average(ts, window)

        # Compute outlier
        outliers = compute_outliers(ts, ma, self.config.get("sensitivity"))

        # Now based on the outliers (true/false), filter the source_df only
        # for the outliers
        source_df = source_df[outliers.index]
        # Remove all columns except timestamp and column
        source_df = source_df[["timestamp", column]]

        return source_df
