import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Engineers technical and business features for the Streaming Churn model.
    
    This class handles data transformation, derived ratio calculations, 
    and missing value imputation based on the 'feature_schema' provided 
    in the configuration.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the engineer with project configuration.
        
        Args:
            cfg (dict): Central configuration dictionary containing 'feature_schema'.
        """
        self.cfg = cfg
        schema = cfg.get("feature_schema", {})
        
        # Core feature groups from model.yaml
        self.num_features = schema.get("numeric", [])
        self.cat_features = schema.get("categorical", [])
        self.target = schema.get("target", "Churned")
        self.excluded = schema.get("excluded", [])

    def get_feature_names(self) -> list:
        """
        Returns the full list of features expected after transformation.
        
        Required by: tests/test_features.py::test_feature_list_consistency
        
        Returns:
            list: Combined list of base and engineered feature names.
        """
        engineered_cols = ["spend_per_month", "is_senior", "Estimated_LTV", "engagement_cost_ratio"]
        return self.num_features + self.cat_features + engineered_cols

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Raw input data.
            
        Returns:
            pd.DataFrame: Transformed data with new features and handled NaNs.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping engineering.")
            return df

        df = df.copy()

        try:
            # 1. Math Ratios & Business Metrics
            df = self._add_business_ratios(df)
            
            # 2. Categorical Binning
            df = self._add_bins(df)
            
            # 3. Data Cleaning & Imputation
            df = self._cleanup_and_impute(df)
            
            return df
        except Exception as e:
            logger.error(f"Critical error during Feature Engineering: {e}")
            raise

    def _add_business_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates LTV and Engagement metrics with zero-division safety."""
        
        # Estimated_LTV logic (Required by: test_create_features_math_logic)
        if "Monthly_Spend" in df.columns and "Subscription_Length" in df.columns:
            df["Estimated_LTV"] = df["Monthly_Spend"] * df["Subscription_Length"]
            
            # Spending velocity
            df["spend_per_month"] = df["Monthly_Spend"] / (df["Subscription_Length"] + 1e-9)

        # Engagement logic (Required by: test_division_by_zero_safety)
        if "Engagement_Score" in df.columns and "Monthly_Spend" in df.columns:
            df["engagement_cost_ratio"] = np.where(
                df["Monthly_Spend"] > 0,
                df["Engagement_Score"] / df["Monthly_Spend"],
                0.0
            )
        else:
            # Fallback if Engagement_Score is missing during tests
            df["engagement_cost_ratio"] = 0.0
            
        return df

    def _add_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates binary indicators for specific demographics."""
        if "Age" in df.columns:
            df["is_senior"] = (df["Age"] >= 60).astype(int)
        return df

    def _cleanup_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles excluded columns and fills missing numeric values."""
        # Drop leaked columns defined in model.yaml
        cols_to_drop = [c for c in self.excluded if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Impute missing values for all numeric features defined in schema
        for col in self.num_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                
        return df
