import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PROCESSED_DATA_PATH

# Imports necessários para o Pipeline do Scikit-Learn (O que faltava!)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataCleaner:
    def __init__(self, missing_threshold=0.5, auto_save=False, output_path=None):
        self.missing_threshold = missing_threshold
        self.auto_save = auto_save
        self.output_path = output_path or PROCESSED_DATA_PATH

    def clean_and_prepare_data(self, df: pd.DataFrame, target_col: str = "Churned") -> tuple:
        """
        Executa a higienização técnica final e separa os tipos de colunas.
        """
        df_clean = df.copy()
        
        # 1. Remoção de Duplicatas e IDs
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        
        cols_to_remove = ["customerID", "Customer_ID", "id", "ID"]
        df_clean = df_clean.drop(columns=[c for c in cols_to_remove if c in df_clean.columns], errors='ignore')
        
        # 2. Tratamento Crítico do Target
        df_clean = df_clean.dropna(subset=[target_col])
        
        # 3. Identificação Dinâmica de Colunas
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_col in num_cols: num_cols.remove(target_col)
        if target_col in cat_cols: cat_cols.remove(target_col)
        
        # 4. Imputação de Segurança (Fallback para limpeza manual)
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna("Unspecified")

        # 5. Persistência
        output_file = Path(self.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_file, index=False)

        print(f"[Quality Assurance] Limpeza concluída: {initial_count} -> {len(df_clean)} registros.")
        return df_clean, num_cols, cat_cols
    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Remove colunas especificadas do DataFrame de forma segura."""
        data = df.copy()
        to_drop = [c for c in columns if c in data.columns]
        return data.drop(columns=to_drop)
    
    def input_missing_values(self, df: pd.DataFrame, num_cols: list, cat_cols: list) -> pd.DataFrame:
            """Imputação manual de dados faltantes (Fallback)."""
            data = df.copy()
            for col in num_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].median())
            for col in cat_cols:
                if col in data.columns:
                    data[col] = data[col].fillna("Unspecified")
            return data
# --- FUNÇÃO EXPORTADA PARA O MAIN.PY ---

def get_preprocessor(num_cols, cat_cols):
    """
    Cria o motor de transformação que será salvo no .joblib
    """
    # Pipeline para dados numéricos
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para dados categóricos
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combina os dois processadores em um ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    return preprocessor

