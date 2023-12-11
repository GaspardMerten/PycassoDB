from datetime import timedelta, datetime

import pandas as pd

__all__ = ["build_period_from_frequency"]


def build_period_from_frequency(
    frequency: str, last_timestamp: pd.Timestamp, before: bool
):
    """
    Build a period from a frequency and a last timestamp. A frequency is [0-9]+[h|d|w|m]. The period is the next/previous
    one from the last timestamp, rounded to the period.
    :param frequency: The frequency
    :param last_timestamp: The last timestamp
    :param before: Whether to get the previous period or the next one
    :return: A period (start, end)
    """

    # Get the period from the frequency
    period = frequency[-1]
    period = {
        "h": "H",
        "d": "D",
        "w": "W",
        "m": "M",
    }[period]

    timedelta_period = {
        "H": timedelta(hours=1),
        "D": timedelta(days=1),
        "W": timedelta(weeks=1),
        "M": timedelta(days=30),
    }[period]

    # Get the number of periods from the frequency
    n_periods = int(frequency[:-1])

    # Get the last timestamp rounded to the period
    last_timestamp = last_timestamp.round(period)

    # Convert last timestamp to a datetime
    last_timestamp = last_timestamp.to_pydatetime()

    # Get the next/previous period
    if before:
        try:
            start = last_timestamp - timedelta_period * n_periods
        except (
            OverflowError
        ):  # If the timestamp is too small, we set it to the minimum datetime
            start = datetime.min
        end = last_timestamp
    else:
        start = last_timestamp
        end = last_timestamp + timedelta_period * n_periods
    # Reconverting to pandas timestamp

    return start, end
