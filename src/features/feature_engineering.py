import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    High-robustness Feature Engineering calibrated for model.yaml schema.
    """
    def __init__(self, cfg: dict):
        """
        Initialization with defensive parsing of the feature_schema.
        """
        self.cfg = cfg
        
        # Mapping exactly to your model.yaml keys
        schema = cfg.get("feature_schema", {})
        self.target = schema.get("target", "Churned")
        self.num_features = schema.get("numeric", [])
        self.cat_features = schema.get("categorical", [])
        self.excluded = schema.get("excluded", [])

        # Business logic from base config (fallback to 12 if not found)
        self.ltv_horizon = cfg.get("base", {}).get("business_logic", {}).get("ltv_horizon_months", 12)

        if not self.num_features:
            logger.error("CRITICAL: Numeric features list is empty in config!")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transformation pipeline with safe execution.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to FeatureEngineer.")
            return df
            
        df = df.copy()
        
        try:
            df = self._add_derived_ratios(df)
            df = self._bin_age(df)
            df = self._cleanup_data(df)
            return df
        except Exception as e:
            logger.error(f"Failed to engineer features: {str(e)}")
            raise

    def _add_derived_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Engagement-to-Spend ratio and LTV metrics.
        """
        # Safe ratio calculation (Engagement_Score / Monthly_Spend)
        if "Engagement_Score" in df.columns and "Monthly_Spend" in df.columns:
            # Using np.where to handle zero-division properly on the whole vector
            df["engagement_cost_ratio"] = np.where(
                df["Monthly_Spend"] > 0, 
                df["Engagement_Score"] / df["Monthly_Spend"], 
                0
            )
        
        return df

    def _bin_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorizes age into generational bins."""
        if "Age" in df.columns:
            df["is_senior"] = (df["Age"] >= 60).astype(int)
        return df

    def _cleanup_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops excluded features and handles remaining NaNs."""
        # Drop leaked or unnecessary columns defined in YAML
        cols_to_drop = [c for c in self.excluded if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Final sanity check: fill NaNs in numeric features
        for col in self.num_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                
        return df
