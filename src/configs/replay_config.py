from __future__ import annotations
from pydantic import BaseModel, Field
from pathlib import Path
from src.utils.io import save_base_model
from src.utils.logging import create_logger
from typing import Any, Dict, List, Optional, Union, Tuple

logger = create_logger(__name__)


class ReplayConfig(BaseModel):
    use_replay: bool = False
    # directory: str|None = None
    grouping_keys: Optional[Union[str, List[str]]] = None
    buffer_size: Optional[int] = None
    buffer_ratio: float = 0.33
    kwargs: dict = Field(default_factory=dict)
    
    def save(self, dir_name: str, file_name: str = "replay_config.json") -> None:
        """
        Save the replay configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)