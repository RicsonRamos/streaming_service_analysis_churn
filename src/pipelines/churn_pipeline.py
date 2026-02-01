import pandas as pd
import joblib
import logging
from pathlib import Path
from src.features.validation import validate_dataframe
from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class ChurnPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fe = FeatureEngineer(cfg)
        self.model_data = None
        self.model_path = Path(cfg["paths"]["models"]["churn_model"])

    def _load_model(self):
        """Carrega o modelo salvo se ele existir."""
        if self.model_path.exists():
            self.model_data = joblib.load(self.model_path)
            return True
        return False

    def process_data(self, df: pd.DataFrame, training=False):
        """O funil único: Valida -> Transforma -> Filtra."""
        # 1. Validação de Contrato
        if not validate_dataframe(df):
            raise ValueError("Dados não passam no contrato do Pydantic.")

        # 2. Feature Engineering
        df_enriched = self.fe.create_features(df)
        
        # 3. Seleção de Colunas (Feature Registry)
        features_list = self.fe.get_feature_names()
        
        if training:
            target = self.cfg["model"]["features"]["target"]
            return df_enriched[features_list], df_enriched[target]
        
        return df_enriched[features_list]

    def predict(self, input_data: pd.DataFrame):
        """Método para usar no Dashboard (Semana 4)."""
        if not self.model_data:
            if not self._load_model():
                raise FileNotFoundError("Modelo não treinado encontrado.")

        # Processa os dados de entrada
        X = self.process_data(input_data, training=False)
        
        # Aplica One-Hot Encoding (alinhado com o treino)
        cat_features = self.cfg["model"]["features"]["categorical"]
        X_final = pd.get_dummies(X, columns=cat_features)
        
        # Garante que as colunas sejam as mesmas do treino (Reindex)
        X_final = X_final.reindex(columns=self.model_data["features"], fill_value=0)
        
        return self.model_data["model"].predict_proba(X_final)[:, 1]