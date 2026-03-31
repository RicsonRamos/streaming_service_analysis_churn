import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Carregador central de configura o nica.

    Respons vel pela leituraa de um arquivo YAML contendo as configura es do projeto.
    """

    def __init__(self, config_file: str = "configs/config.yaml"):
        """Inicializa o carregador com o caminho do arquivo de configura o.

        O caminho relativo do arquivo de configura o  resolvido em rela o ao m dulo atual.
        """
        # Busca a raiz do projeto (3 n veis acima de src/config/loader.py)
        self.config_path = Path(__file__).parent.parent.parent / config_file

    def load(self) -> Dict[str, Any]:
        """Carrega o YAML consolidado.

        Retorna um dicion rio com as configura es do projeto.
        Se o arquivo de configura o n o for encontrado, lan a uma exce o FileNotFoundError.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configura o n o encontrada em: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
