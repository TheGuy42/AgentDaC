from __future__ import annotations
from typing import TypeVar, overload, Literal
from abc import ABC
from pathlib import Path
from pydantic import BaseModel
import re

from src.utils.io import save_base_model, load_base_model


T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel, ABC):
    """
    Base configuration class that provides common functionality.
    All configuration classes should inherit from this class.
    """

    def save(self, dir_name: str | Path, file_name: str | None = None, overwrite: bool = False) -> None:
        """
        Save the configuration to a JSON file.

        Args:
            dir_name: Directory to save the file in.
            file_name: Optional custom filename. If not provided, uses class name.
            overwrite: Whether to overwrite existing file.
        """
        if file_name is None:
            # Generate filename from class name: PromptConfig -> prompt_config.json
            # Convert CamelCase to snake_case
            class_name = self.__class__.__name__
            snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
            file_name = f"{snake_case}.json"

        if isinstance(dir_name, str):
            dir_name = Path(dir_name)

        self.save_to_path(dir_name / file_name, overwrite=overwrite)

    def save_to_path(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save the configuration to a specific path.

        Args:
            path: Full path including filename
            overwrite: Whether to overwrite existing file
        """
        save_base_model(self, path, overwrite=overwrite)

    @overload
    @classmethod
    def load(
        cls: type[T],
        dir_name: str | Path,
        file_name: str | None,
        do_raise: Literal[True],
        **kwargs,
    ) -> T : ...
    
    @overload
    @classmethod
    def load(
        cls: type[T],
        dir_name: str | Path,
        file_name: str | None,
        do_raise: Literal[False],
        **kwargs,
    ) -> T | None: ...

    @classmethod
    def load(
        cls: type[T],
        dir_name: str | Path,
        file_name: str | None = None,
        do_raise: bool = True,
        **kwargs,
    ) -> T | None:
        """
        Load configuration from a JSON file.

        Args:
            dir_name: Directory containing the file.
            file_name: Optional custom filename. If not provided, uses class name.
            do_raise: Whether to raise an error if loading fails, or return None.

        Returns:
            Loaded configuration instance
        """
        if file_name is None:
            # Generate filename from class name
            class_name = cls.__name__
            import re

            snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
            file_name = f"{snake_case}.json"

        if isinstance(dir_name, str):
            dir_name = Path(dir_name)

        return cls.load_from_path(dir_name / file_name, do_raise=do_raise, **kwargs)

    @overload
    @classmethod
    def load_from_path(
        cls: type[T],
        path: str | Path,
        do_raise: Literal[True],
        **kwargs,
    ) -> T : ...
    
    @overload
    @classmethod
    def load_from_path(
        cls: type[T],
        path: str | Path,
        do_raise: Literal[False],
        **kwargs,
    ) -> T | None: ...

    @classmethod
    def load_from_path(
        cls: type[T],
        path: str | Path,
        do_raise=True,
        **kwargs,
    ) -> T | None:
        """
        Load configuration from a specific path.

        Args:
            path: Full path including filename
            do_raise: Whether to raise an error if loading fails, or return None.

        Returns:
            Loaded configuration instance
        """
        return load_base_model(cls, path, do_raise=do_raise, **kwargs)
