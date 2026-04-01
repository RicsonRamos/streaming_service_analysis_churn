"""
Data I/O Module.
Handles all data ingestion and persistence operations.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# 1. INSTANCIAR O LOGGER (O ponto da falha)
logger = logging.getLogger(__name__)

"""
Data I/O Module.
Handles all data ingestion and persistence operations.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# 1. INSTANCIAR O LOGGER (O ponto da falha)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Service for managing data loading and saving operations.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the loader with project configurations.

        Parameters
        ----------
        cfg : dict
            Project configurations.
        """
        self.cfg = cfg

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Garante que os nomes das colunas não tenham espaços ou caracteres invisíveis.
        Rigor: Evita o erro de KeyError que paralisou o pipeline.

        Returns
        -------
        pd.DataFrame
            DataFrame com colunas sanitizadas.
        """
        df.columns = [c.strip() for c in df.columns]
        return df

    def load_raw_data(self) -> pd.DataFrame:
        """
        Carrega os dados brutos e sanitiza os cabeçalhos.

        Returns
        -------
        pd.DataFrame
            DataFrame com dados brutos.
        """
        path = self.cfg["data"]["raw_source"]
        try:
            # Tenta carregar os dados
            df = pd.read_csv(path)

            # Limpeza de nomes de colunas (Rigor Técnico)
            df = self._clean_columns(df)

            logger.info(f"Dados brutos carregados de {path}. Colunas: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Falha crítica ao carregar CSV bruto em {path}: {e}")
            raise e

    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Carrega o dataset processado.

        Returns
        -------
        pd.DataFrame or None
            DataFrame com dados processados ou None se não encontrado.
        """
        path = Path(self.cfg["data"]["final_dataset"])
        if not path.exists():
            logger.error(f"Dados processados não encontrados em: {path}")
            return None

        df = pd.read_csv(path)
        return self._clean_columns(df)

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """
        Persiste o DataFrame no diretório de dados processados.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com dados processados.
        """
        path = Path(self.cfg["data"]["final_dataset"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # Salvamos sem o índice do pandas para não criar colunas 'Unnamed: 0'
            df.to_csv(path, index=False)
            logger.info(f"Dados processados salvos em {path} (Linhas: {len(df)})")
        except Exception as e:
            logger.error(f"Erro ao salvar dados processados em {path}: {e}")
            raise e
