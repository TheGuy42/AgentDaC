"""
Base configuration class to eliminate duplication across config classes.
"""
from __future__ import annotations
from abc import ABC
from pathlib import Path
from pydantic import BaseModel

from src.utils.io import save_base_model


class BaseConfig(BaseModel, ABC):
    """
    Base configuration class that provides common functionality.
    All configuration classes should inherit from this to eliminate duplication.
    """
    
    def save(self, dir_name: str | Path, file_name: str | None = None) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            dir_name: Directory to save the file in
            file_name: Optional custom filename. If not provided, uses class name.
        """
        if file_name is None:
            # Generate filename from class name: PromptConfig -> prompt_config.json
            class_name = self.__class__.__name__
            # Convert CamelCase to snake_case
            import re
            snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
            file_name = f"{snake_case}.json"
        
        if isinstance(dir_name, str):
            dir_name = Path(dir_name)
            
        save_base_model(self, dir_name / file_name)
    
    def save_to_path(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the configuration to a specific path.
        
        Args:
            path: Full path including filename
            overwrite: Whether to overwrite existing file
        """
        save_base_model(self, path, overwrite=overwrite)
    
    @classmethod
    def load(cls, dir_name: str | Path, file_name: str | None = None) -> BaseConfig:
        """
        Load configuration from a JSON file.
        
        Args:
            dir_name: Directory containing the file
            file_name: Optional custom filename. If not provided, uses class name.
            
        Returns:
            Loaded configuration instance
        """
        from src.utils.io import load_base_model
        
        if file_name is None:
            # Generate filename from class name
            class_name = cls.__name__
            import re
            snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
            file_name = f"{snake_case}.json"
        
        if isinstance(dir_name, str):
            dir_name = Path(dir_name)
            
        return load_base_model(cls, dir_name / file_name)
    
    @classmethod
    def load_from_path(cls, path: str | Path) -> BaseConfig:
        """
        Load configuration from a specific path.
        
        Args:
            path: Full path including filename
            
        Returns:
            Loaded configuration instance
        """
        from src.utils.io import load_base_model
        return load_base_model(cls, path)