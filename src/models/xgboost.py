"""
XGBoost Model Wrapper.

Encapsulates the XGBoost classifier with custom logic for training, 
inference, and integration with project-wide configurations.
"""

import joblib
import xgboost as xgb
import pandas as pd
import logging
from typing import Dict, Any, Optional

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
        self.cfg = cfg
        # Extração de hiperparâmetros com fallback seguro
        self.params = cfg.get("hyperparameters", {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        })
        
        # CORREÇÃO: Ativa suporte nativo para colunas categóricas (Pandas category)
        self.params["enable_categorical"] = True
        
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = "logloss"

        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the XGBoost model to the training data.
        """
        logger.info(f"[MODEL] Training XGBoost with params: {self.params}")
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predicts binary classes."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts churn probability for the positive class (1).
        Garante que a ordem das colunas esteja correta antes da inferência.
        """
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns a formatted DataFrame with feature importance scores.
        """
        importance = self.model.feature_importances_
        # Obtém os nomes das features diretamente do booster treinado
        features = self.model.get_booster().feature_names
        
        # Se o booster não tiver nomes (X for numpy), usa índices f0, f1...
        fi_df = pd.DataFrame({
            "feature": features if features else [f"f{i}" for i in range(len(importance))],
            "importance": importance
        }).sort_values(by="importance", ascending=False)
        
        return fi_df

    def save(self, path: str, artifact: dict = None):
        """
        Salva o modelo ou um dicionário de artefatos.
        """
        try:
            if artifact:
                joblib.dump(artifact, path)
                logger.info(f"[INFO] Artefato completo salvo em {path}")
            else:
                # Fallback caso receba apenas o modelo (retrocompatibilidade)
                joblib.dump(self.model, path)
                logger.info(f"[INFO] Apenas o modelo bruto salvo em {path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise e

    def load(self, path: str):
        """
        Loads the full model artifact and restores the internal model object.
        """
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.params = artifact.get("params", self.params)
        return artifact