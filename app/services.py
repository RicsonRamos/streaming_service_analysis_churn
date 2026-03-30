import streamlit as st
import pandas as pd
import joblib
import logging
import mlflow
from src.features.feature_engineering import FeatureEngineer

# 1. INSTANCIAR O LOGGER (O ponto da falha)
logger = logging.getLogger(__name__)

class ChurnService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fe = FeatureEngineer(cfg)
        # Em produção, carregaríamos artefatos reais (medias, scalers) aqui
        self._load_metadata()

    def _load_metadata(self):
        """
        Carrega parâmetros de engenharia calculados no treino.
        Rigor: Garante que o preenchimento de nulos no Dash use os mesmos valores do Treino.
        """
        # Exemplo: self.fe.params = joblib.load('models/artifacts/fe_params.joblib')
        self.fe.params = {'monthly_median': 500.0} 

    def load_model(self):
        """
        Carrega o artefato do modelo. 
        Nota: O st.cache_resource deve ser usado na chamada (main_dash.py) 
        para evitar problemas de serialização aqui.
        """
        try:
            path = self.cfg['artifacts']['current_model']
            artifact = joblib.load(path)
            logger.info(f"Artefato de modelo carregado com sucesso de: {path}")
            return artifact
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo: {e}")
            raise e

    def predict_churn(self, model_artifact, df, threshold):
        """
        Realiza a predição usando o artefato carregado.
        """
        # 1. DESEMPACOTAR O MODELO (Corrigido)
        if isinstance(model_artifact, dict):
            model = model_artifact['model']
            feature_names = model_artifact['features']
        else:
            model = model_artifact
            # Fallback perigoso se feature_names não vier no dict
            feature_names = getattr(model, 'feature_names_in_', None)

        # 2. PRÉ-PROCESSAMENTO (Feature Engineering)
        # X deve conter apenas as colunas que o modelo espera
        X = self.fe.transform(df) 
        
        # 3. ALINHAMENTO E FILTRAGEM DE COLUNAS (Rigor Técnico)
        # Garante que a ordem das colunas e a quantidade sejam idênticas ao treino
        if feature_names is not None:
            # Filtra apenas as colunas que o modelo conhece para evitar erro de shape
            X = X[list(feature_names)]
        else:
            logger.warning("Nomes das features não encontrados. A inferência pode falhar.")

        # 4. INFERÊNCIA
        # O modelo XGBoost agora terá acesso aos dados no formato 'category' ou numérico
        probs = model.predict_proba(X)[:, 1]
        
        # 5. PÓS-PROCESSAMENTO (Visão de Negócio)
        # Criamos uma cópia para não poluir o dataframe original de exibição
        results = df.copy()
        results['Churn_Probability'] = probs
        results['Risk_Level'] = results['Churn_Probability'].apply(
            lambda x: 'High' if x >= threshold else 'Low'
        )
        
        return results