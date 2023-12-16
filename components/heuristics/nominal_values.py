import pandas as pd
from sklearn.cluster import DBSCAN

from src.framework import Component


class NominalValueComponent(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        # Use DBScan to find Outliers in RPM

        # Initialize DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=25)

        outliers = pd.DataFrame()

        # Fit the model and predict
        labels = dbscan.fit_predict(source[['']])

        # Get rows from source that are outliers
        outliers_df = source.iloc[labels == -1]

        # Merge with outliers
