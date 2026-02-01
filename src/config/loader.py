import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, base_path="configs"):
        # Garante que o caminho seja relativo à raiz do projeto, não importa de onde chame
        self.base_path = Path(__file__).parent.parent.parent / base_path

    def _load_yaml(self, path: Path) -> dict:
        """Método interno para ler arquivos YAML."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_all(self) -> dict:
        """
        Carrega dinamicamente todos os arquivos .yaml da pasta configs.
        Se houver base.yaml, paths.yaml e model.yaml, o retorno será:
        { "base": {...}, "paths": {...}, "model": {...} }
        """
        if not self.base_path.exists():
            raise FileNotFoundError(f"Diretório de configuração não encontrado: {self.base_path}")

        full_config = {}
        # Itera sobre todos os arquivos .yaml ou .yml na pasta
        for config_file in self.base_path.glob("*.yaml"):
            key = config_file.stem  # Pega o nome do arquivo sem a extensão
            full_config[key] = self._load_yaml(config_file)
        
        return full_config