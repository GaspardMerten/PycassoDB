import numpy as np
import pandas as pd


def identify_residual_outliers(y, y_pred, std_multiplier=2, ids=None, index=None):
    """
    Identify outliers based on the residuals between actual and predicted values.

    :param y: Array of actual values.
    :param y_pred: Array of predicted values.
    :param std_multiplier: Multiplier for the standard deviation to define the outlier threshold.
    :param ids: Array of ids to be returned with the outliers.
    :param index: Array of indices to be returned with the outliers.
    :return: Array of outlier indices.
    """
    # Calculate residuals
    residuals = np.abs(y - y_pred)

    # Compute the standard deviation of residuals
    std_dev = np.std(residuals)

    # divide residuals by std_dev
    indices = np.abs(residuals // std_dev)

    # Add ids to the outliers (same number of rows as the outliers)
    if ids is not None:
        indices = pd.concat([indices, ids], axis=1)
    if index is not None:
        indices.index = index

    out = pd.DataFrame(indices[indices >= std_multiplier])

    outliers_col = ["outlier_index"]

    if ids is not None:
        outliers_col.append("train_id")

    out.columns = outliers_col
    # Drop where outliers are NaN
    out.dropna(inplace=True)

    return out
