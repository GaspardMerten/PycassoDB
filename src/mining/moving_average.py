def moving_average(ts, window_size, shift=None):
    """
    This function takes a time series as input and returns the moving average
    of the time series, shifted by the given shift. If no shift is given, the
    moving average is shifted by half the window size.
    :param ts: The time series to be averaged
    :param window_size: The size of the window to compute the average
    :param shift: The shift to apply to the moving average
    :return: The moving average of the time series
    """

    return ts.rolling(window_size).mean().shift(shift or window_size // 2)


