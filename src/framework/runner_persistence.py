import json
import os
from datetime import datetime

import pandas as pd


class RunnerPersistence:
    def __init__(self, path: str):
        self.path = path
        self.memory = {}

        if os.path.exists(self.path):
            self.memory = json.load(open(self.path, "r"))

    def register_last_timestamp(
        self, component: str, timestamp: pd.Timestamp, train_id: str = None
    ):
        if train_id:
            self.memory[f"{component}_{train_id}"] = timestamp.isoformat()
        else:
            self.memory[component] = timestamp.isoformat()

        json.dump(self.memory, open(self.path, "w"))

    def get_last_timestamp(self, component: str, train_id: str = None) -> pd.Timestamp:
        if train_id:
            return pd.Timestamp(
                self.memory.get(f"{component}_{train_id}", datetime.min)
            )
        else:
            return pd.Timestamp(self.memory.get(component, datetime.min))
