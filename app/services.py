import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ChurnService:
    def predict_churn(self, model, df, threshold):
        """
        Realiza a predição garantindo alinhamento de tipos para o XGBoost.
        """
        # 1. PRÉ-PROCESSAMENTO (Fiel ao treino)
        drop_cols = ['Churned', 'Customer_ID', 'Satisfaction_Score', 'Last_Activity']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # Garante tratamento de categorias se o XGBoost estiver configurado para tal
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype("category")

        # 2. INFERÊNCIA
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return df

        # 3. PÓS-PROCESSAMENTO
        results = df.copy()
        results['Churn_Probability'] = probs
        results['Risk_Level'] = results['Churn_Probability'].apply(
            lambda x: 'High' if x >= threshold else 'Low'
        )
        return results