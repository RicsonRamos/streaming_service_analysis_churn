"""
ChurnService: Core logic for data loading, inference, and risk classification.
Handles feature engineering alignment between training and production.
"""
import pandas as pd
import joblib
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import shap
from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class ChurnService:
    def __init__(self, model_path: str, processed_path: str, cfg: dict):
        """
        Initializes the service. Now integrates the global config to 
        instantiate the same FeatureEngineer used in training.
        """
        self.model_path = model_path
        self.processed_path = processed_path
        self.cfg = cfg
        # Use the real engine, not a hardcoded copy-paste logic
        self.fe = FeatureEngineer(cfg)

    def load_assets(self) -> Tuple[Optional[Any], Optional[pd.DataFrame]]:
        """Loads model artifacts and the processed historical dataset."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at: {self.model_path}")
            return None, None
            
        if not os.path.exists(self.processed_path):
            logger.error(f"Data file not found at: {self.processed_path}")
            return None, None

        try:
            loaded = joblib.load(self.model_path)
            model = loaded.get('model') if isinstance(loaded, dict) else loaded
            df = pd.read_csv(self.processed_path)
            
            # Extract expected features directly from the trained XGBoost model
            if hasattr(model, 'feature_names_in_'):
                self.model_features = model.feature_names_in_.tolist()
            else:
                # Fallback to config if model doesn't have metadata (not ideal)
                logger.warning("Model lacks feature_names_in_. Falling back to config.")
                self.model_features = self.cfg.get("feature_schema", {}).get("numeric", [])
            
            return model, df
        except Exception as e:
            logger.error(f"Failed to load assets: {e}")
            return None, None

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically aligns the DataFrame with the model's expected features.
        This handles the One-Hot Encoding (Dummies) mismatch.
        """
        # 1. Use the official FeatureEngineer to get business features
        X = self.fe.create_features(df)
        
        # 2. Apply One-Hot Encoding for categorical features defined in config
        cat_cols = self.cfg.get("feature_schema", {}).get("categorical", [])
        X = pd.get_dummies(X, columns=cat_cols)
        
        # 3. SMART ALIGNMENT: 
        # Add missing columns (as 0) and filter/reorder to match the model EXACTLY
        for col in self.model_features:
            if col not in X.columns:
                X[col] = 0
                
        return X[self.model_features]

    def predict_churn(self, model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Performs batch inference while preserving original identification columns."""
        # Align features based on what the model was actually trained on
        X = self._align_features(df)

        # Inference using the cleaned/aligned matrix
        probabilities = model.predict_proba(X)[:, 1]
        
        output_df = df.copy()
        output_df['Probability'] = probabilities
        output_df['Risk_Level'] = output_df['Probability'].apply(
            lambda x: 'High' if x >= threshold else ('Medium' if x >= 0.4 else 'Low')
        )
        
        return output_df

    def predict_single_customer(self, model, customer_data: dict) -> float:
        """Predicts churn probability for a single customer (Simulator)."""
        df_single = pd.DataFrame([customer_data])
        X_single = self._align_features(df_single)
        
        prob = model.predict_proba(X_single)[0, 1]
        return float(prob)

    def get_shap_explanation(self, model, df_input: pd.DataFrame):
        """
        Generates SHAP values with a fallback for version-mismatch errors.
        Ensures the explainer can handle the XGBoost base_score string issue.
        """
        # 1. Alinha as colunas para o formato que o modelo espera
        X_aligned = self._align_features(df_input)
        
        try:
            # Tenta o modo padrão (mais rápido)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_aligned)
            expected_value = explainer.expected_value
        except Exception as e:
            # FALLBACK: Se o XGBoost e o SHAP brigarem pelo base_score (o erro [4.48E-1])
            # Usamos o modo interventional apontando para X_aligned (corrigido)
            explainer = shap.TreeExplainer(
                model, 
                data=X_aligned, 
                feature_perturbation="interventional"
            )
            shap_values = explainer.shap_values(X_aligned)
            expected_value = explainer.expected_value

        return shap_values, X_aligned, expected_value