from __future__ import annotations
from pydantic import BaseModel, Field
from pathlib import Path
from src.utils.io import save_base_model
from src.utils.logging import create_logger
from typing import Any, Dict, List, Optional, Union, Tuple

logger = create_logger(__name__)


class SampleBufferConfig(BaseModel):
    use_buffer: bool = False
    max_size: int = 1000 # 0 for unlimited
    added_ratio: float = 0.33
    
    def save(self, dir_name: str, file_name: str = "sample_buffer_config.json") -> None:
        """
        Save the replay configuration to a JSON file.
        """
        save_base_model(self, Path(dir_name) / file_name)