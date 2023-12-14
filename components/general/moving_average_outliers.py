import pandas as pd

from components.source import SOURCE_DATA_COLS
from src.framework.component import Component
from src.mining.moving_average import moving_average
from src.mining.timeseries_outliers import compute_outliers


class MovingAverageOutlierDetector(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """Run the component."""

        outliers = pd.DataFrame()

        for col in SOURCE_DATA_COLS:
            window = self.config.get("window")

            ts = pd.Series(source[col], index=source.index)

            # Compute moving average
            ma = moving_average(ts, window, shift=1)

            # Compute outlier
            col_outliers = compute_outliers(ts, ma, self.config.get("tolerance"))
            df = pd.DataFrame(col_outliers, columns=[col])
            # Rename col to "value"
            df["type"] = col
            # Merge with outliers
            outliers = pd.concat([outliers, df])

        outliers.sort_index(inplace=True)

        return outliers
