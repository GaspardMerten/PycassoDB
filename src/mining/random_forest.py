import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_random_forest_regressor(X: pd.DataFrame, y: pd.Series):
    """
    Train a Random Forest Regressor using a pipeline and return the trained model.
    :param X: The features
    :param y: The target
    :return: The trained model
    """

    # Create a pipeline with StandardScaler and RandomForestRegressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor())
    ])

    # Train the model using the pipeline
    pipeline.fit(X, y)

    return pipeline
