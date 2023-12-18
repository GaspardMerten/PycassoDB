from datetime import datetime
from typing import List

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

    if "train_id" not in data.columns:
        print("No train_id column in data")
        return None

    # Compute the ranking
    ranking = (
        data.groupby("train_id")[["intensity"]].sum().sort_values(ascending=False, by="intensity")
    )

    return ranking[["intensity"]]


def compute_ranking_for_train_for_outliers_degressive(
    data: pd.DataFrame,
    intensity_column: str = None,
    intensity_mode: str = "multiplicative",
    decay_rate: float = 0.1,
):
    """
    Compute the ranking for all trains based on the number and potential intensity of outliers,
    with more recent outliers being more important.
    :param data: The outliers data for all trains.
    :param intensity_column: (Optional) The column to use for the intensity.
    :param intensity_mode: The mode to use for the intensity. Supported values: multiplicative, additive, exponential.
    :param decay_rate: The rate at which the importance of older outliers decays.
    :return: A DataFrame containing the ranking for all trains.
    """

    # Compute the intensity
    if intensity_column is None or intensity_column not in data.columns:
        data["intensity"] = 1
    else:
        data["intensity"] = data[intensity_column].apply(
            lambda x: _intensity_mode_to_function(intensity_mode)(x, 1)
        )

    # Compute the age of each outlier in days
    current_time = datetime.now()
    # THis line is no longer supported in pandas 1.2.4,
    data["outlier_age"] = data.index.to_series().apply(
        lambda x: (current_time - x).days
    )

    # Apply decay to intensity based on age
    data["adjusted_intensity"] = (
        data["intensity"] * (1 - decay_rate) ** data["outlier_age"]
    )

    if data.empty:
        return None

    # Compute the ranking
    ranking = (
        data.groupby("train_id")["adjusted_intensity"]
        .sum()
        .sort_values(ascending=False)
    )

    return ranking


def combine_rankings(rankings: List[pd.Series]):
    """
    Combine multiple rankings, in order to get a single ranking. The rankings is made by comparing each position in the
    rankings and assigning a score to each train based on its position in the rankings. The score is computed as the
    average position in the rankings.
    :param rankings: The rankings to combine.
    :return: A DataFrame containing the combined ranking.
    """

    # First, compute the rank of each train in each ranking
    ranks = []

    for ranking in rankings:
        if ranking is None:
            continue
        ranks.append(ranking.rank(ascending=True))

    if not ranks:
        return None

    # Then, compute the average rank for each train
    combined_ranking = pd.concat(ranks, axis=1).mean(axis=1)

    # Sort the ranking
    combined_ranking = combined_ranking.sort_values(ascending=False)

    return combined_ranking


def get_ranking_for_components(
    components: List,
    storage_manager: StorageManager,
    period: Period = None,
    decay: bool = False,
) -> pd.Series:
    rankings = []
    for component in components:
        if period is None:
            period = _get_last_30_days(component.name, storage_manager)

        args = dict(
            data=storage_manager.get_for_all_trains(component.name, period),
            intensity_column=component.intensity_column,
            intensity_mode="multiplicative",
        )

        if decay:
            ranking = compute_ranking_for_train_for_outliers_degressive(**args)
        else:
            ranking = compute_ranking_for_train_for_outliers(**args)

        rankings.append(ranking)

    combined_ranking = combine_rankings(rankings)

    return combined_ranking


def _get_last_30_days(component_name, storage_manager):
    try:
        end_timestamp = max(
            {
                storage_manager.get_last_timestamp(component_name, train_id)
                for train_id in storage_manager.retrieve_train_ids()
            }
        )
        period = (end_timestamp - pd.Timedelta(days=30), end_timestamp)
    except ValueError:
        period = None

    return period


def get_outliers_for_train(
    train_id: str,
    component: str,
    storage_manager: StorageManager,
    period: Period = None,
):
    if period is None:
        period = _get_last_30_days(component, storage_manager)

    return storage_manager.get_for_train(component,train_id, period)
