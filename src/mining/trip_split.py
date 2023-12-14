import pandas as pd


__all__ = ["find_trip_split_indexes"]

from matplotlib import pyplot as plt


def find_trip_split_indexes(data: pd.DataFrame, threshold):
    """
    This function takes data as input and splits it into trips time series.
    A trip is defined as a sequence of data with consecutive timestamp values
    that are above the given threshold.

    :param data: The data frame containing timestamps
    :param threshold: The threshold to split the time series (in minutes)
    :return: A list of segments containing index pairs, each segment is a trip
    """
    # TODO: Verify that the data contains a timestamp index

    # Separate data by duration without data
    df = data.copy()

    # Sort the data by timestamp
    df.sort_index(inplace=True)

    # Get the difference between consecutive timestamps
    diff = pd.Series(df.index, index=df.index).diff()

    print(diff)

    # Get the right side indexes where the difference is bigger than 30min
    idx_left = diff[diff > pd.Timedelta(minutes=threshold)].index

    # Get the difference between consecutive timestamps, but in reverse order
    s_diff = df.index.to_series().diff(periods=-1)

    # Get the left side indexes where the difference is bigger than 30min
    idx_right = s_diff[s_diff < -pd.Timedelta(minutes=threshold)].index

    # Add the first and last indexes
    idx_left = [df.index.min()] + idx_left.tolist()
    idx_right = idx_right.tolist() + [df.index.max()]

    return [segment for segment in zip(idx_left, idx_right)]
