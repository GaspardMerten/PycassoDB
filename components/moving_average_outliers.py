import pandas as pd

from src.framework.component import Component
from src.mining.moving_average import moving_average
from src.mining.timeseries_outliers import compute_outliers


class MovingAverageOutlierDetector(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        """Run the component."""
        column = self.config.get("column")
        window = self.config.get("window")

        ts = pd.Series(source[column], index=source.index)

        # Compute moving average
        ma = moving_average(ts, window, shift=1)

        # Compute outlier
        outliers = compute_outliers(ts, ma, self.config.get("sensitivity"))

        return pd.DataFrame(outliers, columns=[column])
