import pandas as pd
from matplotlib import pyplot as plt

from components.source import SOURCE_DATA_COLS
from src.framework.component import Component
from src.mining.moving_average import moving_average
from src.mining.residuals_to_outliers import identify_residual_outliers


class MovingAverageOutlierDetector(Component):
    def run(self, surgery: pd.DataFrame) -> pd.DataFrame:
        print("moving_average")
        """Run the component."""
        outliers = pd.DataFrame()

        for col in SOURCE_DATA_COLS:
            window = self.config.get("window", 5)

            for uuid, trip in surgery.groupby("uuid"):
                ts = pd.Series(trip[col], index=trip.index)

                # Compute moving average
                ma = moving_average(ts, window, shift=1)
                # Compute outlier
                col_outliers = identify_residual_outliers(
                    ts, ma, std_multiplier=4, index=trip.index
                )
                # Rename col to "value"
                col_outliers["type"] = col
                # Merge with outliers
                outliers = pd.concat([outliers, col_outliers])

                if self.debug:
                    # Plot the moving average
                    plt.plot(ts, label="Actual")
                    plt.plot(ma, label="Moving average")
                    plt.legend()
                    plt.show()

        outliers.sort_index(inplace=True)

        return outliers
