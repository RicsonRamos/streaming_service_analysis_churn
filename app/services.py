"""
ChurnService: Core logic for data loading, inference, and risk classification.
Handles feature engineering alignment between training and production.
"""
import streamlit as st
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
        # 1. Gera as features técnicas
        X = self.fe.create_features(df)
        
        # 2. Encoding das categóricas via YAML
        cat_cols = self.cfg.get("feature_schema", {}).get("categorical", [])
        X = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns])
        
        # 3. Alinhamento com as colunas do modelo
        for col in self.model_features:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.model_features]

        # --- A LINHA QUE SALVA SUA VIDA ---
        # Converte tudo para float. Se houver uma string, o Python vai gritar o erro aqui
        # com o nome da coluna culpada, em vez de crashar o SHAP.
        try:
            X = X.apply(pd.to_numeric, errors='raise').astype(float)
        except Exception as e:
            logger.error(f"Erro de conversão de tipo: {e}")
            # Se falhar, vamos identificar quais colunas ainda são 'object'
            bad_cols = X.select_dtypes(include=['object']).columns.tolist()
            raise ValueError(f"As seguintes colunas não são numéricas: {bad_cols}")
            
        return X

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
    
    @st.cache_data
    def get_shap_explanation(_self, _model, df_input: pd.DataFrame): # Adicione _ no self e no model
        """
        Generates SHAP values. The leading underscores in _self and _model 
        prevent Streamlit from trying to hash these complex objects.
        """
        X_aligned = _self._align_features(df_input)
        
        # API Moderna do SHAP (mais estável com XGBoost)
        explainer = shap.Explainer(_model, X_aligned)
        shap_values = explainer(X_aligned)
        
        # O Explainer moderno já traz o expected_value dentro do objeto shap_values
        return shap_values, X_aligned, shap_values.base_values[0]