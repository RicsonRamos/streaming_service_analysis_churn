import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# REMOVIDO: from src.features.feature_engineering import FeatureEngineer (O ERRO ESTAVA AQUI)

class FeatureEngineer:
    """
    Engineers technical and business features for the Streaming Churn model.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        schema = cfg.get("feature_schema", {})
        self.num_features = schema.get("numeric", [])
        self.cat_features = schema.get("categorical", [])

    def get_feature_names(self) -> list:
        engineered = [
            "Estimated_LTV", "Engagement_Score", "LTV_Spend_Ratio", 
            "Is_Free_Trial", "Is_High_Spender", "is_senior"
        ]
        return self.num_features + self.cat_features + engineered

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        try:
            df["Estimated_LTV"] = df["Monthly_Spend"] * df["Subscription_Length"]
            df["Engagement_Score"] = df["Support_Tickets_Raised"] / (df["Subscription_Length"] + 1)
            df["LTV_Spend_Ratio"] = df["Estimated_LTV"] / (df["Monthly_Spend"] + 1e-9)
            df["Is_Free_Trial"] = (df["Subscription_Length"] == 0).astype(int)
            monthly_median = df["Monthly_Spend"].median()
            df["Is_High_Spender"] = (df["Monthly_Spend"] > monthly_median).astype(int)
            df["is_senior"] = (df["Age"] >= 60).astype(int)

            for col in self.num_features:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            return df
        except Exception as e:
            logger.error(f"Feature Engineering failed: {e}")
            raise
