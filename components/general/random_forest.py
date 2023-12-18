import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from src.framework.component import Component
from src.mining.random_forest import train_random_forest_regressor
from src.mining.residuals_to_outliers import identify_residual_outliers

class RandomForestOutliers(Component):
    def run(self, source: pd.DataFrame):
        if source.empty:
            return pd.DataFrame()

        X_columns = self.config.get("X")
        y_column = self.config.get("y")

        assert X_columns is not None, "X columns not specified"
        assert y_column is not None, "y column not specified"

        model = train_random_forest_regressor(source[X_columns], source[y_column])
        # Predict
        y_pred = model.predict(source[X_columns])

        outliers = identify_residual_outliers(
            source[y_column], y_pred, self.config.get("std_multiplier", 4)
        )

        # Inner join with source to get the train_id
        outliers = outliers.merge(source, left_index=True, right_index=True)

        if self.debug:
            print("Number of outliers:", outliers.shape[0])

            # Plot y, y_pred, and outliers
            plt.figure(figsize=(10, 6))
            plt.plot(source.index, source[y_column], label='y', alpha=0.5)
            plt.plot(source.index, source[X_columns[0]], label='x', alpha=0.5)
            plt.scatter(outliers.index, outliers[y_column], c='red', marker='o', label='Outliers')

            plt.xlabel('Data Points')
            plt.ylabel('Values')
            plt.title('Outlier Detection')
            plt.legend()
            plt.show()

        return outliers
