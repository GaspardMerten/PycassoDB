import random

import matplotlib.pyplot as plt
import pandas as pd
# Plot correlation matrix
import seaborn as sns

from src.framework.component import Component


class CorrelDetector(Component):
    def run(self, enriched: pd.DataFrame) -> pd.DataFrame:
        print(len(enriched))
        # Retrieve features
        features_col = self.config.get("features")

        # Plot correlation matrix between features
        corr = enriched[features_col].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.savefig(f"correlation_matrix_{random.randint(1000, 9999)}.png")

        return pd.DataFrame()