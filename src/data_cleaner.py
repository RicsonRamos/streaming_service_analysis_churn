import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display, HTML


class DataCleaner:
    """
    Módulo modular para limpeza, tratamento e imputação de dados.
    """

    def __init__(self, missing_threshold=0.5, auto_save=False, output_path=None):
        self.missing_threshold = missing_threshold
        self.dropped_columns = []
        self.auto_save = auto_save
        self.output_path = output_path

    def _log(self, message):
        display(
            HTML(
                f"<p style='color:#888; font-size:12px; font-family:sans-serif;'>"
                f"🛠️ {message}</p>"
            )
        )

    def impute_columns(self, df: pd.DataFrame, strategy='median', columns=None) -> pd.DataFrame:
        """
        Imputação modular: permite escolher a estratégia e as colunas.
        Estratégias: 'median', 'mean', 'mode', ou um valor fixo.
        """
        data = df.copy()
        cols_to_impute = columns if columns is not None else data.columns

        for col in cols_to_impute:
            if col in data.columns and data[col].isnull().any():
                if strategy == 'median':
                    fill_value = data[col].median()
                elif strategy == 'mean':
                    fill_value = data[col].mean()
                elif strategy == 'mode':
                    fill_value = data[col].mode()[0]
                else:
                    fill_value = strategy  # valor fixo

                data[col] = data[col].fillna(fill_value)
                self._log(f"Coluna '{col}': preenchida com {strategy} ({fill_value})")

        return data
    
    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Remove colunas especificadas do DataFrame."""
        data = df.copy()
        to_drop = [c for c in columns if c in data.columns]
        data = data.drop(columns=to_drop)
        self._log(f"Colunas descartadas: {to_drop}")
        return data

    def clean_data(self, df: pd.DataFrame, auto_impute=True) -> pd.DataFrame:
        """Pipeline principal de limpeza."""
        data = df.copy()

        # 1. Threshold de nulos
        initial_cols = data.columns.tolist()
        limit = len(data) * self.missing_threshold
        data = data.dropna(thresh=limit, axis=1)

        self.dropped_columns = [c for c in initial_cols if c not in data.columns]
        if self.dropped_columns:
            self._log(
                f"Colunas removidas (>{self.missing_threshold*100:.0f}% nulos): "
                f"{self.dropped_columns}"
            )

        # 2. Duplicatas
        dups = data.duplicated().sum()
        if dups > 0:
            data = data.drop_duplicates()
            self._log(f"{dups} duplicatas removidas.")

        # 3. Imputação automática
        if auto_impute:
            num_cols = data.select_dtypes(include=[np.number]).columns
            data = self.impute_columns(data, strategy='median', columns=num_cols)

            cat_cols = data.select_dtypes(include=['object']).columns
            data = self.impute_columns(data, strategy='mode', columns=cat_cols)

        # 4. Salvamento automático (se ativado)
        if self.auto_save and self.output_path:
            save_cleaned_data(data, self.output_path)
            self._log(f"Dados limpos salvos em: {self.output_path}")

        return data

    def remove_outliers_iqr(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Remove outliers usando o método IQR."""
        data = df.copy()

        for col in columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower) & (data[col] <= upper)]

        self._log(f"Outliers removidos (IQR): {columns}")
        return data


def get_cleaning_stats(self, df_before, df_after):
    rows_lost = len(df_before) - len(df_after)
    return {
        "rows_cleaned": rows_lost,
        "retention_rate": (len(df_after) / len(df_before)) * 100,
        "cols_dropped": len(self.dropped_columns)
    }

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Salva o DataFrame limpo no caminho especificado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

