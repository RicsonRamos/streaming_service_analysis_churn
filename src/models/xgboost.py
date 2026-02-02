"""
XGBoost Model Wrapper.

Encapsulates the XGBoost classifier with custom logic for training, 
inference, and integration with project-wide configurations.
"""

import joblib
import xgboost as xgb
import pandas as pd
from typing import Dict, Any, Optional

class ChurnXGBoost:
    """
    High-level wrapper for XGBoost churn classification.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        Initializes the model with parameters from the YAML configuration.

        Args:
            cfg (dict): Global configuration dictionary containing 'hyperparameters'.
        """
        self.cfg = cfg
        # Extract hyperparameters directly from the refactored model.yaml
        self.params = cfg.get("hyperparameters", {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        })
        
        # Adding eval_metric to params if not present
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = "logloss"

        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the XGBoost model to the training data.
        """
        print(f"[MODEL] Training XGBoost with params: {self.params}")
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predicts binary classes."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Predicts churn probability for the positive class (1)."""
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns a formatted DataFrame with feature importance scores.
        """
        importance = self.model.feature_importances_
        features = self.model.get_booster().feature_names
        
        # If booster doesn't have names, it returns 'f0', 'f1', etc.
        # This helper ensures we have a clean DataFrame
        fi_df = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values(by="importance", ascending=False)
        
        return fi_df

    def save(self, path: str, feature_names: list):
        """
        Saves a dictionary artifact containing both the model and the feature list.
        This is crucial to prevent shape mismatch errors in production.
        """
        artifact = {
            "model": self.model,
            "features": feature_names,
            "params": self.params
        }
        joblib.dump(artifact, path)
        print(f"[INFO] Model artifact saved to {path}")

    def load(self, path: str):
        """Loads the full model artifact."""
        artifact = joblib.load(path)
        self.model = artifact["model"]
        return artifact
