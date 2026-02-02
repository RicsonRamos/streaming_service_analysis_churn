"""
Feature Engineering Module.
Transforms raw streaming data into business-relevant features while 
ensuring consistency between training and real-time inference.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Engineers technical and business features for the Streaming Churn model.
    Maintains identity columns and handles missing values for robust inference.
    """
    def __init__(self, cfg: dict):
        """
        Initializes the engineer with schema definitions from config.
        """
        self.cfg = cfg
        schema = cfg.get("feature_schema", {})
        self.num_features = schema.get("numeric", [])
        self.cat_features = schema.get("categorical", [])
        self.target = schema.get("target")
        self.id_col = "Customer_ID"

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point for feature transformation.
        Ensures identification columns are preserved and missing values handled.
        
        Args:
            df (pd.DataFrame): Input dataframe (raw or partially processed).
            
        Returns:
            pd.DataFrame: Enriched dataframe with technical ratios and flags.
        """
        if df.empty:
            return df
            
        X = df.copy()
        
        try:
            # 1. DEFENSIVE CHECK: Ensure basic columns exist for ratios
            # Critical for the Simulator/Inference where input might be sparse
            required_basics = ['Monthly_Spend', 'Subscription_Length', 'Support_Tickets_Raised', 'Age']
            for col in required_basics:
                if col not in X.columns:
                    X[col] = 0.0

            # 2. BUSINESS LOGIC: Ratio & Interaction Features
            # Using 1e-9 or +1 to prevent DivisionByZero errors
            X["Estimated_LTV"] = X["Monthly_Spend"] * X["Subscription_Length"]
            X["Engagement_Score"] = X["Support_Tickets_Raised"] / (X["Subscription_Length"] + 1)
            X["LTV_Spend_Ratio"] = X["Estimated_LTV"] / (X["Monthly_Spend"] + 1e-9)
            X["Is_Free_Trial"] = (X["Subscription_Length"] == 0).astype(int)
            
            # 3. BEHAVIORAL FLAGS
            # Use median from config or current batch to determine high spenders
            monthly_median = X["Monthly_Spend"].median() if not X["Monthly_Spend"].empty else 0
            X["Is_High_Spender"] = (X["Monthly_Spend"] > monthly_median).astype(int)
            X["is_senior"] = (X["Age"] >= 60).astype(int)

            # 4. MISSING VALUE IMPUTATION
            # Only impute columns that are actually present
            for col in self.num_features:
                if col in X.columns:
                    X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)

            # 5. IDENTITY PRESERVATION
            if self.id_col not in X.columns:
                logger.debug(f"Note: {self.id_col} not in dataframe. Proceeding without ID.")

            return X
            
        except Exception as e:
            logger.error(f"Feature Engineering transformation failed: {e}")
            raise

    def get_model_feature_names(self) -> list:
        """
        Returns a list of all features intended for the model (excluding ID and Target).
        Useful for documentation and schema validation.
        """
        engineered = [
            "Estimated_LTV", "Engagement_Score", "LTV_Spend_Ratio", 
            "Is_Free_Trial", "Is_High_Spender", "is_senior"
        ]
        return self.num_features + self.cat_features + engineered