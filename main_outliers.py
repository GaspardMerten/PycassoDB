import logging

import pandas as pd
from matplotlib import pyplot as plt

from src.framework import (
    load_config_from_file,
    StorageManager,
)
from src.framework.outliers import (
    compute_ranking_for_train_for_outliers,
    compute_ranking_for_train_for_outliers_degressive,
    combine_rankings,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load the global and component configuration
    config = load_config_from_file("config.toml")
    # Instantiate the storage manager, which handles all data storage for the components
    storage_manager = StorageManager(config.storage_folder)

    rankings = []

    for name, component_config in config.components.items():
        if not component_config.outliers_producer:
            continue

        start_timestamp = min(
            {
                storage_manager.get_first_timestamp(name, train_id)
                for train_id in storage_manager.retrieve_train_ids()
            }
        )
        end_timestamp = max(
            {
                storage_manager.get_last_timestamp(name, train_id)
                for train_id in storage_manager.retrieve_train_ids()
            }
        )

        period = (end_timestamp - pd.Timedelta(days=30), end_timestamp)

        ranking = compute_ranking_for_train_for_outliers(
            data=storage_manager.get_for_all_trains(name, period),
            intensity_column=component_config.intensity_column,
            intensity_mode="multiplicative",
        )

        progressive_ranking = compute_ranking_for_train_for_outliers_degressive(
            data=storage_manager.get_for_all_trains(name, period),
            intensity_column=component_config.intensity_column,
            intensity_mode="multiplicative",
        )

        rankings.append(progressive_ranking)

        # Plot the ranking
        progressive_ranking.head(10).plot.bar()
        plt.show()

    combined_ranking = combine_rankings(rankings)
    # Plot the ranking
    combined_ranking.head(100).plot.bar()
    plt.title("Combined ranking")
    plt.show()
