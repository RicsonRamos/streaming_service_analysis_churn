import pandas as pd
import os

from src.config.loader import ConfigLoader

cfg = ConfigLoader().load_all()

RAW_PATH = cfg["paths"]["data"]["raw"]
PROCESSED_PATH = cfg["paths"]["data"]["processed"]
MODEL_PATH = cfg["paths"]["models"]["churn_model"]


def load_streaming_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carrega e aplica limpezas básicas (Sanity Check)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Exemplo de lógica de negócio: Monthly Spend não pode ser negativo
    if 'Monthly_Spend' in df.columns:
        df['Monthly_Spend'] = df['Monthly_Spend'].clip(lower=0)
        
    return df

def get_data_metrics(df: pd.DataFrame):
    """Calcula métricas sem renderizar HTML."""
    metrics = {
        'rows': df.shape[0],
        'cols': df.shape[1],
        'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'duplicates': df.duplicated().sum(),
        'null_info': (df.isnull().sum() / len(df) * 100).round(2)
    }
    return metrics