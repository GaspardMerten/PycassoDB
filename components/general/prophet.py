import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet

from src.framework.component import Component


def _prepare_for_prophet(source, y_column):
    source = source[[y_column]]
    # rename y_column to y
    source = source.rename(columns={y_column: "y"})
    # transform index to ds
    source = source.reset_index().rename(columns={"timestamp": "ds", "index": "ds"})
    return source


class ProphetOutliers(Component):
    def run(
        self,
        source: pd.DataFrame,
        source_before: pd.DataFrame,
    ):
        if source.empty or source_before.empty:
            return pd.DataFrame()

        y_column = self.config.get("y")
        assert y_column is not None, "y column not specified"

        source_before = _prepare_for_prophet(source_before, y_column)

        model = Prophet(changepoint_prior_scale=0.01, yearly_seasonality=True).fit(
            source_before
        )

        # Predict
        forecast = model.make_future_dataframe(
            periods=14 * 24, freq="H", include_history=False
        )

        forecast = model.predict(forecast)

        # group source by train_id
        source = source.groupby("train_id")

        for train_id, train in source:
            # Compute the median of the train for each hour for y
            train = train.groupby(train.index.hour).median()

            # Now if median is 30% different from the forecast, then it is an outlier
            train["yhat"] = forecast["yhat"]

            train["outlier"] = train["yhat"].apply(
                lambda x: True
                if abs(x - train[y_column].iloc[0]) > 0.3 * train[y_column].iloc[0]
                else False
            )

            # Compute the residuals
            train["residual"] = train["yhat"] - train[y_column].iloc[0]

            # Compute the z-score
            train["z_score"] = (train["residual"] - train["residual"].mean()) / train["residual"].std()

            # Compute the outliers
            train["outlier"] = train["z_score"].apply(lambda x: True if abs(x) > 3 else False)

            # Plot everything
            plt.plot(train["yhat"], label="yhat")
            plt.plot(train[y_column], label="y")
            plt.plot(train["residual"], label="residual")
            plt.plot(train["z_score"], label="z_score")
            plt.plot(train["outlier"], label="outlier")
            plt.legend()
            plt.show()

        return pd.DataFrame()
