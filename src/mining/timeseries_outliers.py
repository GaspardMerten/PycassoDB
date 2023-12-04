import pandas as pd


def compute_outliers(ts1: pd.Series, ts2: pd.Series, tolerance: int) -> pd.Series:
    """
    This function takes two time series as input and returns the outliers of the
    :param ts1: The first time series
    :param ts2: The second time series
    :param tolerance: The tolerance of the outlier detection
    :return: The outliers of the difference between the two time series
    """

    diff = ts1 - ts2
    # Compute derivative of difference
    diff = diff.diff()
    # Absolute value of derivative
    diff = diff.abs()

    # Compute mean and standard deviation of derivative
    mean = diff.mean()
    std = diff.std()

    # Compute outliers
    outliers = diff[diff > mean + tolerance * std]

    return outliers
