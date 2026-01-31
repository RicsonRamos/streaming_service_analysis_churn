import yaml
from pathlib import Path


class ConfigLoader:

    def __init__(self, base_path="configs"):
        self.base_path = Path(base_path)

    def load_yaml(self, name: str) -> dict:
        path = self.base_path / name

        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_all(self) -> dict:
        base = self.load_yaml("base.yaml")
        paths = self.load_yaml("paths.yaml")

        return {
            "base": base,
            "paths": paths
        }
