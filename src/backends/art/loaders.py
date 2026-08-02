import socket
from contextlib import closing

import art
from art.local import LocalBackend

from src.backends.art.config import ModelConfig
from src.backends.art.paths import PathConfig
from src.utils.logging import create_logger


logger = create_logger(__name__)


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


async def load_art_model(
    path_config: PathConfig,
    model_config: ModelConfig,
    port: int | None = None,
    seed: int | None = None,
) -> art.TrainableModel:
    if path_config.base_model != model_config.base_model:
        raise ValueError(f"Model name mismatch: {path_config.base_model} != {model_config.base_model}.")

    if port is None:
        port = find_free_port()
        logger.info(f"Found free port for ART server: {port}")

    model_config = model_config.initialize(output_dir=path_config.model_output_dir, port=port, seed=seed)

    model = art.TrainableModel(
        name=path_config.run_name,
        project=path_config.project_name,
        base_model=path_config.base_model,
        _internal_config=model_config.internal_config,
    )

    backend = LocalBackend(path=path_config.art_path, in_process=False)
    await model.register(backend, _openai_client_config=model_config.openai_config)
    return model
