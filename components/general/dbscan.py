import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.framework.component import Component


class DBScanOutliers(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        # Initialize DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=25)  # adjust these parameters as needed

        source_features = source[self.config.get("features")].values

        # Normalize the data
        source_features = StandardScaler().fit_transform(source_features)

        # Fit the model and predict
        labels = dbscan.fit_predict(source_features)

        # Get rows from source that are outliers
        outliers = source.iloc[labels == -1]

        if self.debug:
            # Plot the clusters
            plt.scatter(
                source_features[:, 0], source_features[:, 1], c=labels, cmap="plasma"
            )
            plt.legend()
            plt.show()

        return outliers
