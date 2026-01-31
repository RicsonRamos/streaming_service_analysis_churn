import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DATA_PATH # Importando o caminho do seu config

class FeatureEngineer:
    def __init__(self):
        self.original_cols = []
        self.final_cols = []

    def _log(self, msg):
        print(f"[Feature Engineering] {msg}")

    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        cols = data.columns
        
        # 1. Mapeamento exato baseado no seu log de colunas
        # LTV = Duração da Assinatura * Gasto Mensal
        tenure_col = 'Subscription_Length'
        spend_col = 'Monthly_Spend'
        
        # Engajamento = Score de Satisfação / (Tickets de Suporte + 1)
        # Nota: Usamos Satisfação como numerador porque quanto maior, melhor o engajamento
        sat_col = 'Satisfaction_Score'
        support_col = 'Support_Tickets_Raised'

        # Criando KPI de Valor (LTV)
        if tenure_col in cols and spend_col in cols:
            data['Estimated_LTV'] = data[tenure_col] * data[spend_col]
            self._log(f"KPI Criado: Estimated_LTV (Baseado em {tenure_col})")

        # Criando KPI de Comportamento (Engagement)
        if sat_col in cols and support_col in cols:
            data['Engagement_Score'] = data[sat_col] / (data[support_col] + 1)
            self._log(f"KPI Criado: Engagement_Score (Baseado em {sat_col})")

        return data

    def process(self, df: pd.DataFrame, target='Churned', drop_cols=['Customer_ID'], save=True):
        """Pipeline completo com opção de salvamento."""
        self.original_cols = df.columns.tolist()
        
        # 1. Enriquecimento
        data = self.create_business_features(df)

        # 2. Limpeza
        to_drop = [c for c in drop_cols if c in data.columns]
        data = data.drop(columns=to_drop)
        self._log(f"Colunas descartadas: {to_drop}")

        # 3. Encoding
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if target in cat_cols:
            cat_cols.remove(target)
        data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

        # 4. Tipagem do Target
        if target in data.columns:
            data[target] = data[target].astype(int)

        self.final_cols = data.columns.tolist()

        # 5. SALVAMENTO (A parte que faltava)
        if save:
            output_path = Path(PROCESSED_DATA_PATH)
            output_path.parent.mkdir(parents=True, exist_ok=True) # Cria a pasta 'processed' se não existir
            data.to_csv(output_path, index=False)
            self._log(f"Dataset final (Gold) salvo em: {output_path}")

        self._log(f"Pipeline finalizado. Features totais: {len(self.final_cols)}")
        return data