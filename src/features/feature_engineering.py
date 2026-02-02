import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_features = cfg["model"]["features"]["numeric"]
        self.cat_features = cfg["model"]["features"]["categorical"]

    def create_features(self, df):
        """
        Cria novas variáveis para aumentar o poder preditivo (Recall).
        """
        df = df.copy()

        # --- PASSO 1: CRIAR COLUNAS BASE (O alicerce) ---
        # Estas colunas precisam ser criadas PRIMEIRO porque as outras dependem delas.
        df['Estimated_LTV'] = df['Monthly_Spend'] * df['Subscription_Length']
        df['Engagement_Score'] = df['Support_Tickets_Raised'] / (df['Subscription_Length'] + 1)

        # --- PASSO 2: CRIAR COLUNAS DERIVADAS (O telhado) ---
        # Agora sim o 'Estimated_LTV' e 'Engagement_Score' existem.
        
        # 1. Eficiência de Gasto (LTV Ratio)
        df['LTV_Spend_Ratio'] = df['Estimated_LTV'] / (df['Monthly_Spend'] + 1)

        # 2. Score de Engajamento por Posse (Tenure)
        df['Engagement_per_Month'] = df['Engagement_Score'] / (df['Subscription_Length'] + 1)

        # 3. Tickets por Engajamento
        df['Ticket_Engagement_Ratio'] = df['Support_Tickets_Raised'] / (df['Engagement_Score'] + 1)

        # --- PASSO 3: BINARIZAÇÃO (As flags) ---
        df['Is_High_Spender'] = (df['Monthly_Spend'] > df['Monthly_Spend'].median()).astype(int)
        df['Is_Inactive'] = (df['Engagement_Score'] == 0).astype(int)
        df['Is_Free_Trial'] = (df['Monthly_Spend'] == 0).astype(int)
        
        return df

def get_feature_names(self):
        """Retorna a lista completa e ordenada de colunas para o modelo."""
        # Colunas que você criou no Passo 1 e Passo 2
        engineered_features = [
            'Estimated_LTV',           # Faltava esta!
            'Engagement_Score',        # Faltava esta!
            'LTV_Spend_Ratio', 
            'Engagement_per_Month', 
            'Ticket_Engagement_Ratio',
            'Is_High_Spender',
            'Is_Inactive',             # Recomendo incluir se o modelo as viu no treino
            'Is_Free_Trial'            # Recomendo incluir se o modelo as viu no treino
        ]
        
        # O modelo espera: Numericas Originais + Engenharia + Categoricas Originais
        return self.num_features + engineered_features + self.cat_features