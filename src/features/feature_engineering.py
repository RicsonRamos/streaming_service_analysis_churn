import pandas as pd
import numpy as np
import logging

# Setup basic logging to replace print statements
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Robust Feature Engineering class with schema validation and error handling.
    """
    def __init__(self, cfg: dict):
        """
        Initializes with defensive configuration loading.
        """
        self.cfg = cfg
        # Safe access using .get() to prevent KeyErrors during initialization
        model_cfg = cfg.get("model", {})
        feature_cfg = model_cfg.get("features", {})
        
        self.num_features = feature_cfg.get("numeric", [])
        self.cat_features = feature_cfg.get("categorical", [])
        
        # Mandatory business logic parameters with fallbacks
        biz_cfg = cfg.get("base", {}).get("business_logic", {})
        self.ltv_horizon = biz_cfg.get("ltv_horizon_months", 12)

        if not self.num_features:
            logger.warning("No numeric features found in configuration!")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline for feature creation.
        """
        df = df.copy()
        
        try:
            df = self._add_ratios(df)
            df = self._bin_age(df)
            df = self._handle_missing(df)
            return df
        except Exception as e:
            logger.error(f"Critical error during feature engineering: {e}")
            raise

    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates LTV to Spend ratio with zero-division protection.
        """
        # Protect against missing columns
        if "Monthly_Spend" in df.columns and "Subscription_Length" in df.columns:
            # Avoid division by zero by adding a tiny epsilon or using np.where
            df["ltv_spend_ratio"] = (df["Monthly_Spend"] * df["Subscription_Length"]) / (df["Monthly_Spend"] + 1e-9)
        else:
            logger.error("Required columns for ratios are missing from DataFrame")
            
        return df

    def _bin_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bins age into categories with boundary handling.
        """
        if "Age" in df.columns:
            bins = [0, 18, 30, 45, 60, 120]
            labels = ["minor", "young_adult", "adult", "senior", "elderly"]
            df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute remaining NaNs to prevent model crashes.
        """
        for col in self.num_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        return df
