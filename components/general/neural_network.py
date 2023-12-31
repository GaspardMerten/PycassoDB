import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier

from src.framework import Component
from src.mining.residuals_to_outliers import identify_residual_outliers


class NeuralNetworkOutliers(Component):
    def run(
        self, enriched_before: pd.DataFrame, enriched: pd.DataFrame
    ) -> pd.DataFrame:
        print("Launching Neural Network Outlier Detection")
        X = self.config.get("X")
        y = self.config.get("y")

        enriched_before = enriched_before[[*X, y]]
        enriched = enriched[[*X, y, "train_id"]]

        splits = self.config.get("splits", [(0, len(enriched_before))])
        trained_models = []

        # Train different models for each split (split of the before dataset)
        for min_index, max_index in splits:
            print("Training model for split: ", min_index, max_index)
            # Get the training data
            X_train = enriched_before.iloc[min_index:max_index][X]
            y_train = enriched_before.iloc[min_index:max_index][y]

            # Train the model
            model = MLPClassifier(
                hidden_layer_sizes=(100, 100),
                max_iter=500,
                activation="relu",
                solver="adam",
                verbose=True,
            )
            model.fit(X_train, y_train)
            trained_models.append(model)

        # Predict the outliers using the different models
        outliers = pd.DataFrame()
        preds = []
        for model in trained_models:
            # Predict y for the enriched dataset
            y_pred = model.predict(enriched[X])
            preds.append(y_pred)
            outliers = pd.concat(
                [
                    identify_residual_outliers(
                        enriched[y],
                        y_pred,
                        ids=enriched[["train_id"]],
                        index=enriched.index,
                    ),
                    outliers,
                ]
            )

        if self.debug:
            # Group per train_id and plot y_pred vs y
            for train_id, group in enriched.groupby("train_id"):
                plt.plot(
                    group.index,
                    group[y],
                    label="True",
                )
                for model in trained_models:
                    plt.plot(
                        group.index,
                        model.predict(group[X]),
                        label="Predictions",
                    )
                plt.title(train_id)
                plt.show()
        return outliers
