from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os

class Scenario(ABC):
    """
    Abstract base class for a DoomGen scenario.
    A Scenario encapsulates the logic for generating a specific type of map
    based on a provided configuration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: str = "scenario"):
        """
        Initialize the scenario with a configuration.

        Args:
            config: Dictionary of parameters for generation.
            name: Base name for the scenario (used for output filenames).
        """
        self.config = self.get_default_config()
        if config:
            self.config.update(config)
        self.name = name
        self.validate_config()

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Returns the default configuration dictionary.
        Subclasses must implement this.
        """
        pass

    @abstractmethod
    def generate(self, output_path: str) -> None:
        """
        Generates the WAD file for this scenario configuration.

        Args:
            output_path: The full path to write the WAD file.
        """
        pass

    def validate_config(self) -> None:
        """
        Validates the current configuration.
        Can be overridden by subclasses to add specific checks.
        """
        pass

    def get_param(self, key: str) -> Any:
        """Helper to get a config parameter safely."""
        return self.config.get(key)
