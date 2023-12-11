import json
import os
from datetime import datetime

import pandas as pd

__all__ = ["RunnerPersistence"]

class RunnerPersistence:
    """
    A class for managing persistence of runner's state.

    Attributes:
        path (str): The file path for storing the runner's state.
        lock: A lock object for thread-safe operations (default is None).
        memory (dict): In-memory storage of the runner's state.

    Methods:
        update_memory(): Updates the in-memory state from the file.
        register_last_timestamp(component, timestamp, train_id): Registers the last timestamp of a component.
        get_did_run(component, train_id): Checks if a component has run.
        get_last_timestamp(component, train_id): Retrieves the last timestamp of a component.
        acquire_for_lock(): Acquires the lock if it exists.
        release_lock(): Releases the lock if it exists.
    """

    def __init__(self, path: str, lock=None):
        """
        Initializes the RunnerPersistence object.
        """
        self.path = path
        self.lock = lock
        self.memory = {}
        self.update_memory()

    def update_memory(self):
        """Updates the in-memory state from the file."""
        if os.path.exists(self.path):
            with open(self.path, "r") as file:
                self.memory = json.load(file)

    def register_last_timestamp(
        self, component: str, timestamp: pd.Timestamp, train_id: str = None
    ):
        """
        Registers the last timestamp for a specified component.
        """
        self.acquire_for_lock()
        self.update_memory()
        key = f"{component}_{train_id}" if train_id else component
        self.memory[key] = timestamp.isoformat()

        with open(self.path, "w") as file:
            json.dump(self.memory, file)

        self.release_lock()

    def get_did_run(self, component: str, train_id: str = None) -> bool:
        """
        Checks if a component has run.
        """
        key = f"{component}_{train_id}" if train_id else component
        return self.memory.get(key, False)

    def get_last_timestamp(self, component: str, train_id: str = None) -> pd.Timestamp:
        """
        Retrieves the last timestamp for a specified component.
        """
        key = f"{component}_{train_id}" if train_id else component
        return pd.Timestamp(self.memory.get(key, datetime.min))

    def acquire_for_lock(self):
        """Acquires the lock if it exists."""
        if self.lock:
            self.lock.acquire()

    def release_lock(self):
        """Releases the lock if it exists."""
        if self.lock:
            self.lock.release()
