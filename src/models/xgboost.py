"""
XGBoost Model Wrapper.

Encapsulates the XGBoost classifier with custom logic for training,
inference, and integration with project-wide configurations.
"""

import logging
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


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
        self.cfg: Dict[str, Any] = cfg
        self.params: Dict[str, Any] = cfg.get(
            "hyperparameters",
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            },
        )

        self.params["enable_categorical"] = True
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = "logloss"

        self.model: xgb.XGBClassifier = xgb.XGBClassifier(**self.params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fits the XGBoost model to the training data.
        """
        logger.info(f"[MODEL] Training XGBoost with params: {self.params}")
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts binary classes.

        Returns:
            pd.Series: Binary predictions (0/1) for the test set.
        """
        return pd.Series(self.model.predict(X), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts churn probability for the positive class (1).

        Returns:
            pd.Series: Churn probability predictions for the test set.
        """
        return pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns a formatted DataFrame with feature importance scores.

        Returns:
            pd.DataFrame: DataFrame with feature importance scores.
        """
        importance = self.model.feature_importances_
        features = self.model.get_booster().feature_names

        fi_df = pd.DataFrame(
            {
                "feature": features if features else [f"f{i}" for i in range(len(importance))],
                "importance": importance,
            }
        ).sort_values(by="importance", ascending=False)

        return fi_df

    def save(self, path: str, artifact: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves the model or a dictionary of artifacts.

        Args:
            path (str): Path to store the model artifact.
            artifact (dict, optional): Dictionary of artifacts to save alongside the model.
        """
        try:
            if artifact is not None:
                joblib.dump(artifact, path)
                logger.info(f"[INFO] Complete artifact saved at {path}")
            else:
                joblib.dump(self.model, path)
                logger.info(f"[INFO] Only the raw model saved at {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise e

    def load(self, path: str) -> Any:
        """
        Loads the full model artifact and restores the internal model object.

        Args:
            path (str): Path to load the model artifact.

        Returns:
            Any: Loaded artifact dictionary or model.
        """
        artifact: Any = joblib.load(path)
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.params = artifact.get("params", self.params)
        else:
            self.model = artifact  # backward compatibility
        return artifact