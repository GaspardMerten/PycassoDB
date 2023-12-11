import pandas as pd

from src.framework import Period, StorageManager


def get_outliers_per_train_for_component(
    component: str, period: Period, storage_manager: StorageManager
):
    """
    Get the outliers for a component for a period.
    :param component: The component.
    :param period: The period.
    :param storage_manager: The storage manager.
    :return: A DataFrame containing the outliers for the component for the period.
    """
    return storage_manager.get_for_all_trains(component, period)


def _intensity_mode_to_function(intensity_mode: str):
    """
    Convert the intensity mode to a function.
    :param intensity_mode: The intensity mode.
    :return: The function corresponding to the intensity mode.
    """
    if intensity_mode == "multiplicative":
        return lambda x, y: x * y
    elif intensity_mode == "additive":
        return lambda x, y: x + y
    elif intensity_mode == "exponential":
        return lambda x, y: x**y
    else:
        raise ValueError("Intensity mode not supported")


def compute_ranking_for_train_for_outliers(
    data: pd.DataFrame,
    intensity_column: str = None,
    intensity_mode: str = "multiplicative",
):
    """
    Compute the ranking for all trains based on the number and potential intensity of outliers.
    :param data: The outliers data for all trains.
    :param intensity_column: (Optional) The column to use for the intensity.
    :param intensity_mode: The mode to use for the intensity. Supported values: multiplicative, additive, exponential.
    :return: A DataFrame containing the ranking for all trains.
    """

    # If intensity column is not specified, use the number of outliers as intensity
    if intensity_column is None:
        data["intensity"] = 1
    else:
        # Compute the intensity
        data["intensity"] = data[intensity_column].apply(
            lambda x: _intensity_mode_to_function(intensity_mode)(x, 1)
        )

    # Compute the ranking
    ranking = (
        data.groupby("train_id").sum().sort_values(by="intensity", ascending=False)
    )

    return ranking


import pandas as pd
from datetime import datetime


def compute_ranking_for_train_for_outliers_degressive(
    data: pd.DataFrame,
    intensity_column: str = None,
    intensity_mode: str = "multiplicative",
    timestamp_column: str = "timestamp",
    decay_rate: float = 0.1,
):
    """
    Compute the ranking for all trains based on the number and potential intensity of outliers,
    with more recent outliers being more important.
    :param data: The outliers data for all trains.
    :param intensity_column: (Optional) The column to use for the intensity.
    :param intensity_mode: The mode to use for the intensity. Supported values: multiplicative, additive, exponential.
    :param timestamp_column: The column that contains the timestamp of the outliers.
    :param decay_rate: The rate at which the importance of older outliers decays.
    :return: A DataFrame containing the ranking for all trains.
    """

    # Ensure that the timestamp column is a datetime object
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])

    # Compute the intensity
    if intensity_column is None:
        data["intensity"] = 1
    else:
        data["intensity"] = data[intensity_column].apply(
            lambda x: _intensity_mode_to_function(intensity_mode)(x, 1)
        )

    # Compute the age of each outlier in days
    current_time = datetime.now()
    data["outlier_age"] = (current_time - data[timestamp_column]).dt.days

    # Apply decay to intensity based on age
    data["adjusted_intensity"] = (
        data["intensity"] * (1 - decay_rate) ** data["outlier_age"]
    )

    # Compute the ranking
    ranking = (
        data.groupby("train_id")["adjusted_intensity"]
        .sum()
        .sort_values(ascending=False)
    )

    return ranking
