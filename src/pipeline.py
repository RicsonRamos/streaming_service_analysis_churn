import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from IPython.display import display, HTML

from src.config.loader import ConfigLoader

cfg = ConfigLoader().load_all()

RANDOM_STATE = cfg["base"]["runtime"]["random_state"]
TEST_SIZE = cfg["base"]["runtime"]["test_size"]

RAW_PATH = cfg["paths"]["data"]["raw"]
PROCESSED_PATH = cfg["paths"]["data"]["processed"]
MODEL_PATH = cfg["paths"]["models"]["churn_model"]

class ChurnPipeline:
    """
    Pipeline de Machine Learning focado em interpretabilidade e KPIs de negócio.
    """
    def __init__(self, target_col="Churned"):
        self.target_col = target_col
        self.model_name = None
        self.full_pipeline = None

    def _log_step(self, message):
        print(f"✅ [ML Pipeline] {message}")

    def build_and_split(self, df: pd.DataFrame):
        """Prepara os dados e cria o preprocessor."""
        # 1. Limpeza de Segurança
        df = df.dropna(subset=[self.target_col])
        
        # 2. X/y Split
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].map({'Yes': 1, 'No': 0}) if df[self.target_col].dtype == object else df[self.target_col]

        # 3. Identificação Automática de Atributos
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        self._log_step(f"Features identificadas: {len(num_cols)} numéricas, {len(cat_cols)} categóricas.")

        # 4. Construção do Preprocessor (Engine técnica)
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ])

        # 5. Split Estratificado (Mantém a proporção de churn em treino e teste)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        return preprocessor, X_train, X_test, y_train, y_test

    def train_and_report(self, model, df: pd.DataFrame):
        """Treina o modelo e exibe o relatório focado em KPIs para acionistas."""
        preprocessor, X_train, X_test, y_train, y_test = self.build_and_split(df)
        self.model_name = model.__class__.__name__

        # Criar Pipeline Integro
        self.full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Treino
        self._log_step(f"Iniciando treinamento do modelo: {self.model_name}")
        self.full_pipeline.fit(X_train, y_train)

        # Avaliação de Negócio
        y_pred = self.full_pipeline.predict(X_test)
        auc = roc_auc_score(y_test, self.full_pipeline.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else 0
        
        self._display_business_metrics(y_test, y_pred, auc)
        
        return self.full_pipeline

    def _display_business_metrics(self, y_test, y_pred, auc):
        """Traduz métricas técnicas (F1-Score) para visão de acionistas."""
        cm = confusion_matrix(y_test, y_pred)
        # Verdadeiros Positivos (Clientes que íamos perder e o modelo avisou)
        saved_potential = cm[1, 1] 
        # Falsos Negativos (Clientes que saíram e o modelo não viu - O prejuízo real)
        missed_churn = cm[1, 0]

        html = f"""
        <div style="padding: 20px; background-color: #1a1a1a; border-radius: 10px; border-left: 8px solid #448aff; margin: 20px 0;">
            <h3 style="color: #ffffff; margin-top: 0;">📈 Performance Preditiva: {self.model_name}</h3>
            <p style="color: #888;"><b>Poder de Discriminação (AUC):</b> {auc:.2%}</p>
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1; background: #2e7d32; padding: 15px; border-radius: 5px;">
                    <small>ALERTAS DE RETENÇÃO</small><br>
                    <b style="font-size: 20px;">{saved_potential}</b> clientes detectados em risco.
                </div>
                <div style="flex: 1; background: #c62828; padding: 15px; border-radius: 5px;">
                    <small>CHURN NÃO DETECTADO</small><br>
                    <b style="font-size: 20px;">{missed_churn}</b> falhas de previsão.
                </div>
            </div>
            <p style="font-size: 11px; color: #666; margin-top: 10px;">*Baseado em um conjunto de teste de {len(y_test)} clientes.</p>
        </div>
        """
        display(HTML(html))
        print("Relatório Técnico Detalhado:")
        print(classification_report(y_test, y_pred))