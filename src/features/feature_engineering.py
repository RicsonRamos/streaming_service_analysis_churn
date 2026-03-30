import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.schema = cfg.get("feature_schema", {})
        self.target = self.schema.get("target")
        self.drop_cols = self.schema.get("drop_columns", [])
        self.params = {} 

    def fit(self, X: pd.DataFrame):
        """
        Calcula estatísticas APENAS nos dados de treino.
        Rigor: Evita que informações do teste vazem para o treino via médias/medianas.
        """
        logger.info("Calculando parâmetros de Feature Engineering (Fit)...")
        
        # Exemplo: Salvar medianas para preenchimento de nulos futuro
        if 'MonthlyCharges' in X.columns:
            self.params['median_monthly'] = X['MonthlyCharges'].median()
        
        # Salvar categorias conhecidas para evitar erros de novos níveis na inferência
        cat_features = self.schema.get("categorical", [])
        self.params['categories'] = {
            col: X[col].unique().tolist() for col in cat_features if col in X.columns
        }
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica as transformações de forma determinística.
        """
        X = df.copy()
        
        # 1. REMOÇÃO DE COLUNAS PROIBIDAS (Anti-Leakage)
        # Além do target (que o pipeline remove), precisamos tirar o que foi definido no YAML
        cols_to_drop = [c for c in self.drop_cols if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            logger.info(f"Colunas removidas no transform: {cols_to_drop}")

        # 2. TRATAMENTO DE NULOS (Usando parâmetros do Fit)
        if 'MonthlyCharges' in X.columns and 'median_monthly' in self.params:
            X['MonthlyCharges'] = X['MonthlyCharges'].fillna(self.params['median_monthly'])

        # 3. CODIFICAÇÃO DE CATEGÓRICAS (XGBoost Native)
        cat_features = self.schema.get("categorical", [])
        for col in cat_features:
            if col in X.columns:
                # Converte para category e garante que tipos não-previstos virem NaN
                X[col] = pd.Categorical(X[col], categories=self.params.get('categories', {}).get(col))
                X[col] = X[col].astype("category")

        # 4. FILTRAGEM FINAL DE TIPOS
        # Mantém apenas numéricas e categorias. Remove strings (object) que o XGBoost rejeita.
        X = X.select_dtypes(include=['number', 'category'])
        
        # 5. ALINHAMENTO COM O TARGET
        # Se o target ainda estiver no DF (caso do treino), ele deve ser mantido para o pipeline dar o pop()
        if self.target in df.columns and self.target not in X.columns:
            X[self.target] = df[self.target]

        return X