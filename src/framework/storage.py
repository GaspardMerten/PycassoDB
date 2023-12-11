import json
import math
import os
from datetime import datetime
from typing import Tuple, Union, List, Iterable

import pandas as pd
import pyarrow

__all__ = ["StorageManager", "Period"]

Period = Tuple[Union[pd.Timestamp, None], Union[pd.Timestamp, None]]


def _read_rows_from_files(files, limit):
    df = pd.DataFrame()
    for index, file in enumerate(files):
        try:
            df = pd.concat([df, pd.read_parquet(file)])
            if len(df) >= (limit or math.inf):
                df = df[:limit]
                break
        except pyarrow.lib.ArrowInvalid:
            pass

    return df


class StorageManager:
    """
    A class for managing storage of data.

    Attributes:
        path (str): The file path for storing the data.
        _cached_train_ids (set): A set of cached train ids.

    Methods:
        _update_train_ids(train_ids): Updates the cached train ids and the train_ids.json file.
        _append_df_to_parquet(filename, df): Appends new data to an existing parquet file.
        _validate_index(data): Ensures that the index is a DateTimeIndex.
        _store_agnostic(date, group, name): Stores data in a single file (data not related to a specific train).
        _store_per_train(date, data, name, train_id): Stores data in a separate file for each train.
        store(data, name, train_id): Stores data in parquet files, optionally grouping by train_id.
        _list_files(directory, period, invert): Lists all files in a directory, optionally filtering by a date period.
        slice_df_with_period(df, period): Slices a dataframe with a period.
        get_for_train(name, train_id, period, limit, invert): Gets data for a specific train_id, optionally filtering by a date period.
        get_for_all_trains(name, period, limit, invert): Gets data for all trains, optionally filtering by a date period.
        get_for_agnostic(name, period, limit, invert): Gets data not related to a specific train, optionally filtering by a date period.
        get_first_timestamp(name, train_id): Gets the first timestamp for a specific train_id or for all trains.
        get_last_timestamp(name, train_id): Gets the last timestamp for a specific train_id or for all trains.
        has_sufficient_data_since(timestamp, amount, name, train_id): Checks if there is sufficient data since a given timestamp for a specific train_id or for all trains.
        retrieve_train_ids(): Retrieves all train_ids.
    """

    def __init__(self, path):
        self.path = path
        self._cached_train_ids = None

    def _update_train_ids(self, train_ids: set):
        """Update cached train_ids and train_ids.json"""

        train_ids_file = f"{self.path}/train_ids.json"

        if os.path.exists(train_ids_file):
            self._cached_train_ids = set(
                json.load(open(f"{self.path}/train_ids.json", "r"))
            )
        else:
            self._cached_train_ids = set()

        # Check if there are new train_ids
        new_train_ids = train_ids - self._cached_train_ids

        if new_train_ids:
            # Update cached train_ids
            self._cached_train_ids = self._cached_train_ids.union(new_train_ids)
            # Update train_ids.json
            json.dump(list(self._cached_train_ids), open(train_ids_file, "w"))

    @staticmethod
    def _append_df_to_parquet(filename, df):
        """Append new data to existing parquet file"""
        if os.path.exists(filename):
            current = pd.read_parquet(filename)
            # Concatenate current and new data
            df = pd.concat([current, df])
            # Drop duplicate dates
            df = df[~df.index.duplicated(keep="last")]

        df.to_parquet(filename)

    @staticmethod
    def _validate_index(data):
        """Ensure index is DateTimeIndex"""

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                "Index must be DateTimeIndex, use df.set_index('timestamp', inplace=True)"
            )

        # Make sure index is sorted
        if not data.index.is_monotonic_increasing:
            raise ValueError("Index must be sorted, use df.sort_index(inplace=True)")

    def _store_agnostic(self, date, group, name):
        """Store data in a single file, (data not related to a specific train)"""

        file = f"{self.path}/{name}/{date.date()}.parquet"

        self._append_df_to_parquet(file, group)

    def _store_per_train(self, date, data, name, train_id=None):
        """Store data in a separate file for each train_id"""

        def _inner(train, df):
            # Drop train_id column if present
            if "train_id" in df.columns:
                df = df.drop(columns=["train_id"])

            self._update_train_ids({train})
            os.makedirs(f"{self.path}/{name}/{train}", exist_ok=True)
            # Read current file if it exists
            filename = f"{self.path}/{name}/{train}/{date.date()}.parquet"
            self._append_df_to_parquet(filename, df)

        if train_id:
            _inner(train_id, data)
        else:
            for train_id, sub_group in data.groupby("train_id"):
                _inner(train_id, sub_group)

    def store(self, data: pd.DataFrame, name: str, train_id=None):
        """
        Store data in parquet files, optionally grouping by train_id.
        The data is stored in a directory named after the name parameter.
        If train_id is present in the data, the data is stored in a subdirectory named after the train_id.
        The data is grouped by day, each day is stored in a separate file.

        :param data: The data to store, a DataFrame with a DateTimeIndex
        :param name: The name of the dataset
        :param train_id: Optional, the train_id to group by
        """

        if data.empty:
            return

        self._validate_index(data)

        os.makedirs(f"{self.path}/{name}", exist_ok=True)

        # Store each group in a separate file
        for date, group in data.groupby(data.index.floor("d")):
            date: pd.Timestamp

            # If train_id is present, group by train_id, store each group in a separate file
            if "train_id" in data.columns or train_id:
                self._store_per_train(date, group, name, train_id)
            else:
                self._store_agnostic(date, group, name)

    @staticmethod
    def _list_files(
        directory: str,
        period: Period = None,
        invert: bool = False,
    ) -> Iterable[str]:
        """List all files in a directory, optionally filtering by a date period."""
        if not os.path.exists(directory):
            return []

        files = os.listdir(directory)

        if period:
            start_date, end_date = period
            start_date = start_date or datetime.min
            end_date = end_date or datetime.max

            files = [
                file
                for file in files
                if start_date <= datetime.strptime(file, "%Y-%m-%d.parquet") <= end_date
            ]

        files.sort(
            key=lambda file: datetime.strptime(file, "%Y-%m-%d.parquet"), reverse=invert
        )

        return list(os.path.join(directory, file) for file in files)

    @staticmethod
    def slice_df_with_period(df, period: Period = None) -> pd.DataFrame:
        """Slice a dataframe with a period"""
        if period:
            start_date, end_date = period

            # Use index (RangeIndex) to slice
            if isinstance(df.index, pd.RangeIndex):
                df = df[df.index >= start_date.timestamp()]
                df = df[df.index <= end_date.timestamp()]

        return df

    def get_for_train(
        self,
        name: str,
        train_id: str,
        period: Period = None,
        limit: int = None,
        invert: bool = False,
    ) -> pd.DataFrame:
        """Get data for a specific train_id, optionally filtering by a date period."""
        files = self._list_files(f"{self.path}/{name}/{train_id}", period, invert)
        df = _read_rows_from_files(files, limit)

        return self.slice_df_with_period(df, period)

    def get_for_all_trains(
        self, name: str, period: Period = None, limit: int = None, invert: bool = False
    ) -> pd.DataFrame:
        """Get data for all trains, optionally filtering by a date period."""

        df = pd.DataFrame()

        for train_id in self.retrieve_train_ids():
            train_df = self.get_for_train(name, train_id, period, limit, invert)
            train_df["train_id"] = int(train_id)
            df = pd.concat([df, train_df])

        # Sort by timestamp
        df.sort_index(inplace=True)

        # Keep only first limit rows
        if limit:
            df = df[:limit]

        return self.slice_df_with_period(df, period)

    def get_for_agnostic(
        self, name: str, period: Period = None, limit: int = None, invert: bool = False
    ) -> pd.DataFrame:
        """Get data not related to a specific train, optionally filtering by a date period."""
        files = self._list_files(f"{self.path}/{name}", period, invert)

        df = _read_rows_from_files(files, limit)

        return self.slice_df_with_period(df, period)

    def get_first_timestamp(self, name: str, train_id: str = None) -> pd.Timestamp:
        """Get the first timestamp for a specific train_id or for all trains"""
        if train_id:
            files = self._list_files(f"{self.path}/{name}/{train_id}", invert=True)
        else:
            files = self._list_files(f"{self.path}/{name}", invert=True)

        if not files:
            return pd.Timestamp(datetime.max)

        # Get first file
        first_file = list(files)[-1]

        # Read first file
        df = pd.read_parquet(first_file)
        # Return first timestamp
        return df.index.min()

    def get_last_timestamp(self, name: str, train_id: str = None) -> pd.Timestamp:
        """Get the last timestamp for a specific train_id or for all trains"""
        if train_id:
            files = self._list_files(f"{self.path}/{name}/{train_id}")
        else:
            files = self._list_files(f"{self.path}/{name}")

        if not files:
            return pd.Timestamp(datetime.min)

        # Get last file
        last_file = list(files)[-1]

        # Read last file
        df = pd.read_parquet(last_file)
        # Return last timestamp
        return df.index.max()

    def has_sufficient_data_since(
        self, timestamp: pd.Timestamp, amount: int, name: str, train_id: str = None
    ) -> bool:
        """Check if there is sufficient data since a given timestamp for a specific train_id or for all trains"""
        if train_id:
            files = self._list_files(
                f"{self.path}/{name}/{train_id}", (timestamp, None)
            )
        else:
            files = self._list_files(f"{self.path}/{name}", (timestamp, None))

        count = 0
        is_first = True

        for file in files:
            df = pd.read_parquet(file)

            if is_first:
                df = df[df.index >= timestamp]

            count += len(df)
            if count >= (amount or 0):
                return True
        if count >= (amount or 0):
            return True
        return False

    def retrieve_train_ids(self) -> List[str]:
        """Retrieve all train_ids"""
        self._update_train_ids(set())

        return list(self._cached_train_ids)
