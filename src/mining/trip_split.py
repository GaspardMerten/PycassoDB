import pandas as pd


def find_trip_split_indexes(data: pd.DataFrame, threshold):
    """
    This function takes data as input and splits it into trips time series.
    A trip is defined as a sequence of data with consecutive timestamp values
    that are above the given threshold.

    :param data: The data frame containing timestamps
    :param threshold: The threshold to split the time series (in minutes)
    :return: A list of segments containing index pairs, each segment is a trip
    """
    # Verify that the data contains a timestamp column
    if 'timestamps_UTC' not in data.columns:
        raise ValueError('The data does not contain a timestamp column')

    # Sort the data by timestamp
    data = data.sort_values(by='timestamps_UTC')

    # Separate data by duration without data
    data_sample = data.copy()

    # Reset the index
    data_sample = data_sample.reset_index(drop=True)

    # Get the difference between consecutive timestamps
    diff = data_sample['timestamps_UTC'].diff()

    # Get the right side indexes where the difference is bigger than 30min
    idx_left = diff[diff > pd.Timedelta(minutes=threshold)].index

    # Get the left side indexes where the difference is bigger than 30min
    idx_right = idx_left - 1

    # Add the first and last indexes
    idx_left = [0] + idx_left.tolist()
    idx_right = idx_right.tolist() + [len(data_sample)]

    return [segment for segment in zip(idx_left, idx_right)]
