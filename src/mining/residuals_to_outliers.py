import numpy as np
import pandas as pd


def identify_residual_outliers(y, y_pred, std_multiplier=1):
    """
    Identify outliers based on the residuals between actual and predicted values.

    :param y: Array of actual values.
    :param y_pred: Array of predicted values.
    :param std_multiplier: Multiplier for the standard deviation to define the outlier threshold.
    :return: Array of outlier indices.
    """
    # Calculate residuals
    residuals = np.abs(y - y_pred)

    # Compute the standard deviation of residuals
    std_dev = np.std(residuals)

    # divide residuals by std_dev
    indices = residuals // std_dev

    out = pd.DataFrame(indices[indices >= std_multiplier])
    out.columns = ["outlier_index"]

    return out
