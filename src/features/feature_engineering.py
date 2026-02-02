"""
Feature Engineering Module.

Transforms raw input data into enriched features optimized for 
XGBoost classification. Handles feature creation, scaling logic, 
and maintains feature consistency.
"""

import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    """
    Engineers new features from base customer data to improve model recall.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the engineer with schema from configuration.

        Args:
            cfg (dict): Global configuration dictionary.
        """
        self.cfg = cfg
        # Updated to match the new schema in model.yaml
        self.num_features = cfg["feature_schema"]["numeric"]
        self.cat_features = cfg["feature_schema"]["categorical"]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering pipeline.

        Args:
            df (pd.DataFrame): Input dataframe with base columns.

        Returns:
            pd.DataFrame: Enriched dataframe with engineered features.
        """
        df = df.copy()

        # --- STEP 1: BASE METRICS ---
        # Using +1 smoothing to avoid division by zero errors
        df['Estimated_LTV'] = df['Monthly_Spend'] * df['Subscription_Length']
        df['Engagement_Score'] = df['Support_Tickets_Raised'] / (df['Subscription_Length'] + 1)

        # --- STEP 2: DERIVED RATIOS ---
        # LTV vs Spend efficiency
        df['LTV_Spend_Ratio'] = df['Estimated_LTV'] / (df['Monthly_Spend'] + 1)

        # Normalized engagement by tenure
        df['Engagement_per_Month'] = df['Engagement_Score'] / (df['Subscription_Length'] + 1)

        # Behavioral indicators
        df['Ticket_Engagement_Ratio'] = df['Support_Tickets_Raised'] / (df['Engagement_Score'] + 0.1)

        # --- STEP 3: BEHAVIORAL FLAGS ---
        # Median-based spending flag
        spend_median = df['Monthly_Spend'].median()
        df['Is_High_Spender'] = (df['Monthly_Spend'] > spend_median).astype(int)
        
        # Zero-engagement flag
        df['Is_Inactive'] = (df['Engagement_Score'] == 0).astype(int)
        
        # Logic for trial or promotional accounts
        df['Is_Free_Trial'] = (df['Monthly_Spend'] == 0).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """
        Returns the complete, ordered list of features required by the model.

        Returns:
            List[str]: Combined list of numeric, engineered, and categorical features.
        """
        engineered_features = [
            'Estimated_LTV',
            'Engagement_Score',
            'LTV_Spend_Ratio', 
            'Engagement_per_Month', 
            'Ticket_Engagement_Ratio',
            'Is_High_Spender',
            'Is_Inactive',
            'Is_Free_Trial'
        ]
        return self.num_features + engineered_features + self.cat_features
