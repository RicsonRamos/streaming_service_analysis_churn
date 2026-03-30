import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Carregador central de configuração única."""
    def __init__(self, config_file: str = "configs/config.yaml"):
        # Busca a raiz do projeto (3 níveis acima de src/config/loader.py)
        self.config_path = Path(__file__).parent.parent.parent / config_file

    def load(self) -> Dict[str, Any]:
        """Carrega o YAML consolidado."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuração não encontrada em: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}