"""
Data I/O Module.

Handles all data ingestion and persistence operations, ensuring consistency 
between disk storage and memory DataFrames.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

class DataLoader:
    """
    Service for managing data loading and saving operations.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the loader with project configurations.

        Args:
            cfg (dict): Global configuration dictionary.
        """
        self.cfg = cfg

    def load_raw_data(self) -> Optional[pd.DataFrame]:
        """
        Loads the raw streaming dataset from the configured path.

        Returns:
            pd.DataFrame: The raw data or None if file is missing.
        """
        path = Path(self.cfg["data"]["raw_source"])
        if not path.exists():
            print(f"[ERROR] Raw data not found at {path}")
            return None
        return pd.read_csv(path)

    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Loads the clean, feature-engineered dataset for training/inference.

        Returns:
            pd.DataFrame: The processed data or None if file is missing.
        """
        path = Path(self.cfg["data"]["final_dataset"])
        if not path.exists():
            print(f"[ERROR] Processed data not found at {path}")
            return None
        return pd.read_csv(path)

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Persists a DataFrame to the processed data directory.

        Args:
            df (pd.DataFrame): DataFrame to be saved.
        """
        path = Path(self.cfg["data"]["final_dataset"])
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[INFO] Processed data saved to {path}")
