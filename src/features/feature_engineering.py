import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Engineers features for the churn model, ensuring consistency with model.yaml.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        schema = cfg.get("feature_schema", {})
        self.num_features = schema.get("numeric", [])
        self.cat_features = schema.get("categorical", [])
        self.target = schema.get("target", "Churned")
        self.excluded = schema.get("excluded", [])

    def get_feature_names(self) -> list:
        """
        Returns the exact list of features expected by the test.
        The test expects 15 columns: (4 numeric + 8 categorical) + 3 engineered.
        Adjust this logic to match your specific YAML schema.
        """
        engineered = ["spend_per_month", "is_senior", "Estimated_LTV"]
        return self.num_features + self.cat_features + engineered

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline. Protects against missing 'Engagement_Score' during tests.
        """
        df = df.copy()
        
        try:
            # Derived Business Metrics
            if "Monthly_Spend" in df.columns and "Subscription_Length" in df.columns:
                df["Estimated_LTV"] = df["Monthly_Spend"] * df["Subscription_Length"]
                df["spend_per_month"] = df["Monthly_Spend"] / (df["Subscription_Length"] + 1e-9)
            
            # Engagement logic (only if column exists, to prevent KeyError in tests)
            if "Engagement_Score" in df.columns and "Monthly_Spend" in df.columns:
                df["engagement_cost_ratio"] = df["Engagement_Score"] / (df["Monthly_Spend"] + 1e-9)

            # Categorical Bins
            if "Age" in df.columns:
                df["is_senior"] = (df["Age"] >= 60).astype(int)

            # Cleanup: Fill NaNs and drop excluded
            df = self._final_cleanup(df)
            return df
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.excluded if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        for col in self.num_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        return df
