import logging

import pandas as pd

logger = logging.getLogger(__name__)


class ChurnService:
    """
    Serviço responsável por realizar predições de churn em conjuntos de dados.
    """

    def predict_churn(self, model, df, threshold):
        """
        Realiza a predição garantindo alinhamento de tipos para o XGBoost.

        Parameters
        ----------
        model : XGBoost
            Modelo treinado para predição de churn
        df : pandas.DataFrame
            Conjunto de dados com as features a serem utilizadas para predição
        threshold : float
            Nível de risco (probabilidade) acima do qual o cliente é considerado 'High Risk'

        Returns
        -------
        pandas.DataFrame
            Conjunto de dados com as features originais e as colunas adicionais 'Churn_Probability' e 'Risk_Level'
        """
        # 1. PRÉ-PROCESSAMENTO (Fiel ao treino)
        drop_cols = [
            "Churned",
            "Customer_ID",
            "Satisfaction_Score",
            "Last_Activity",
        ]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Garante tratamento de categorias se o XGBoost estiver configurado para tal
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category")

        # 2. INFERÊNCIA
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return df

        # 3. PÓS-PROCESSAMENTO
        results = df.copy()
        results["Churn_Probability"] = probs
        results["Risk_Level"] = results["Churn_Probability"].apply(
            lambda x: "High" if x >= threshold else "Low"
        )
        return results
