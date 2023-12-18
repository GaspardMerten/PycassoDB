import pandas as pd

__all__ = ["compute_ts_residuals"]


def compute_ts_residuals(ts1: pd.Series, ts2: pd.Series) -> pd.Series:
    return ts1 - ts2
