"""
Configuration Loader Module.

Centralizes the loading of YAML configuration files from the project's 
config directory, merging them into a single accessible dictionary.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """
    Handles dynamic loading of YAML configuration files.
    """

    def __init__(self, config_dir: str = "configs/config"):
        """
        Initializes the loader with the path to the configuration directory.

        Args:
            config_dir (str): Relative path to the yaml files.
        """
        # Resolves the absolute path based on the project root
        self.base_path = Path(__file__).parent.parent.parent / config_dir

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """
        Reads a YAML file and returns its content as a dictionary.

        Args:
            path (Path): Path to the target YAML file.

        Returns:
            Dict: Content of the file or empty dict if fails.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                return content if content is not None else {}
        except Exception as e:
            print(f"[ERROR] Failed to load config file {path}: {e}")
            return {}

    def load_all(self) -> Dict[str, Any]:
        """
        Dynamically loads all .yaml files from the config directory.
        
        It merges all files into a single dictionary to facilitate access 
        (e.g., cfg['artifacts'] instead of cfg['paths']['artifacts']).

        Returns:
            Dict: A merged dictionary containing all configuration keys.

        Raises:
            FileNotFoundError: If the config directory does not exist.
        """
        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Configuration directory not found at: {self.base_path.absolute()}"
            )

        full_config = {}
        # Iterates through all .yaml files in the directory
        for config_file in self.base_path.glob("*.yaml"):
            file_content = self._load_yaml(config_file)
            
            # Merges the content directly into the main dictionary
            # This prevents deep nested keys like cfg['model']['model']['type']
            full_config.update(file_content)

        return full_config
