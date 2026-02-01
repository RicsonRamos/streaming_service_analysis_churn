import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_features = cfg["model"]["features"]["numeric"]
        self.cat_features = cfg["model"]["features"]["categorical"]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria novas variáveis para aumentar o poder preditivo (Recall).
        """
        df = df.copy()

        # 1. Eficiência de Gasto (LTV Ratio)
        # Se o LTV estimado for baixo em relação ao gasto mensal, o risco aumenta.
        df['LTV_Spend_Ratio'] = df['Estimated_LTV'] / (df['Monthly_Spend'] + 1)

        # 2. Score de Engajamento por Posse (Tenure)
        # Clientes novos com baixo engajamento são churn iminente.
        df['Engagement_per_Month'] = df['Engagement_Score'] / (df['Subscription_Length'] + 1)

        # 3. Tickets por Engajamento
        # Muitos tickets com pouco uso da plataforma = Frustração.
        df['Ticket_Engagement_Ratio'] = df['Support_Tickets_Raised'] / (df['Engagement_Score'] + 1)

        # 4. Binarização de Comportamento Crítico (Flags)
        df['Is_High_Spender'] = (df['Monthly_Spend'] > df['Monthly_Spend'].median()).astype(int)

        df['Is_Inactive'] = (df['Engagement_Score'] == 0).astype(int)

        df['Is_Free_Trial'] = (df['Monthly_Spend'] == 0).astype(int)
        
        return df

    def get_feature_names(self):
        """Retorna a lista atualizada de colunas para o modelo."""
        new_features = [
            'LTV_Spend_Ratio', 
            'Engagement_per_Month', 
            'Ticket_Engagement_Ratio',
            'Is_High_Spender'
        ]
        return self.num_features + new_features + self.cat_features