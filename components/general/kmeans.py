import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from components.source import SOURCE_DATA_COLS
from src.framework.component import Component


class KMeansOutliers(Component):
    def run(self, source: pd.DataFrame) -> pd.DataFrame:
        os.environ["LOKY_MAX_CPU_COUNT"] = "4"
        # Initialize KMeans
        clusters = self.config.get("clusters", 10)
        kmeans = KMeans(n_clusters=clusters, n_init="auto")

        source_features = source[self.config.get("features", SOURCE_DATA_COLS)].copy()

        source_features = StandardScaler().fit_transform(source_features)

        # Fit the model
        kmeans.fit(source_features)

        # Compute outliers as one belonging to any cluster with less than 5% of the total points
        outliers = pd.DataFrame()

        for i in range(clusters):
            cluster = source[kmeans.labels_ == i]
            if cluster.shape[0] < source.shape[0] * 0.05:
                outliers = pd.concat([outliers, cluster])

        outliers.sort_index(inplace=True)

        if self.debug:
            # Plot the clusters
            plt.scatter(
                source_features[:, 0],
                source_features[:, 1],
                c=kmeans.labels_,
                cmap="plasma",
            )
            plt.show()
            print(f"Found {outliers.shape[0]} outliers")

        return outliers
