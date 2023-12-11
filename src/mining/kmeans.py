from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_kmeans_model(source_data, n_clusters=10, n_init="auto"):
    """
    Build and fit a KMeans model.

    :param source_data: DataFrame containing the source data.
    :param n_clusters: Number of clusters to form.
    :param n_init: Number of time the k-means algorithm will be run with different centroid seeds.
    :return: Fitted KMeans model.
    """
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)

    # Normalize the data
    source_features = StandardScaler().fit_transform(source_data)

    # Fit the model
    kmeans.fit(source_features)

    return kmeans
