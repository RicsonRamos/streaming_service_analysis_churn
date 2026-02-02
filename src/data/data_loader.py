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
    Includes automated identity generation for traceability.
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
        Loads the raw streaming dataset and ensures a unique Customer_ID exists.

        Returns:
            pd.DataFrame: The raw data with an identity column.
        """
        path = Path(self.cfg["data"]["raw_source"])
        if not path.exists():
            print(f"[ERROR] Raw data not found at {path}")
            return None
        
        df = pd.read_csv(path)

        # BRAIN CHECK: Se o ID não existe, criamos AGORA.
        # Isso garante que o ID nasça na ingestão e siga por toda a pipeline.
        if 'Customer_ID' not in df.columns:
            print("[INFO] 'Customer_ID' not found in raw source. Generating sequential IDs...")
            df.insert(0, 'Customer_ID', [f"CUST-{i+1:05d}" for i in range(len(df))])
        
        return df

    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Loads the clean, feature-engineered dataset for training/inference.
        """
        path = Path(self.cfg["data"]["final_dataset"])
        if not path.exists():
            print(f"[ERROR] Processed data not found at {path}")
            return None
        return pd.read_csv(path)

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Persists a DataFrame to the processed data directory.
        The ID generated in load_raw_data will be saved physically here.
        """
        path = Path(self.cfg["data"]["final_dataset"])
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvamos SEM o index do pandas, mas COM a coluna Customer_ID que criamos
        df.to_csv(path, index=False)
        print(f"[INFO] Processed data saved to {path} (Rows: {len(df)})")