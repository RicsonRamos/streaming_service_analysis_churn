import pandas as pd
import joblib
import os
import streamlit as st

class ChurnService:
    def __init__(self, model_path, processed_path):
        self.model_path = model_path
        self.processed_path = processed_path
        # As 17 colunas exatas que o seu modelo exige
        self.expected_features = [
            'Age', 'Subscription_Length', 'Support_Tickets_Raised', 'Satisfaction_Score', 
            'Last_Activity', 'Monthly_Spend', 'Estimated_LTV', 'Engagement_Score',
            'Gender_Female', 'Gender_Male', 
            'Region_Central', 'Region_East', 'Region_North', 'Region_South', 'Region_West',
            'Payment_Method_Credit Card', 'Payment_Method_PayPal'
        ]

    def load_assets(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.processed_path):
            return None, None
        
        loaded = joblib.load(self.model_path)
        model = loaded.get('model') if isinstance(loaded, dict) else loaded
        df = pd.read_csv(self.processed_path)
        return model, df

    def predict_churn(self, model, df, threshold):
        # 1. Preparação
        X = df.copy()
        X = pd.get_dummies(X, columns=['Gender', 'Region', 'Payment_Method'])
        
        # 2. Alinhamento Forçado (Solução do Shape Mismatch 17 colunas)
        for col in self.expected_features:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.expected_features]
        
        # 3. Inferência
        probs = model.predict_proba(X)[:, 1]
        df['Probabilidade'] = probs
        df['Nivel_Risco'] = df['Probabilidade'].apply(
            lambda x: 'Alto' if x >= threshold else ('Médio' if x >= 0.4 else 'Baixo')
        )
        return df
    
    def get_feature_importance(self, model):
        """
        Extrai a importância das colunas do modelo XGBoost.
        """
        import pandas as pd
        
        # Pega as importâncias e associa aos nomes das 17 colunas
        importances = model.feature_importances_
        feature_names = self.expected_features
        
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        return fi_df
    def predict_single_customer(self, model, customer_data):
        """
        Recebe um dicionário com os dados de 1 cliente e retorna a probabilidade.
        """
        import pandas as pd
        
        # Converte para DataFrame
        X_single = pd.DataFrame([customer_data])
        
        # Aplica o mesmo One-Hot Encoding do treino
        X_single = pd.get_dummies(X_single)
        
        # Alinhamento forçado com as 17 colunas
        for col in self.expected_features:
            if col not in X_single.columns:
                X_single[col] = 0
        
        X_single = X_single[self.expected_features]
        
        # Retorna a probabilidade (0 a 1)
        return model.predict_proba(X_single)[0, 1]