import socket
from contextlib import closing
import art
from art.local import LocalBackend
from src.configs import PathConfig, VllmConfig, ArtConfig
from src.utils.logging import create_logger


logger = create_logger(__name__)


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


async def load_art_model(
    path_config: PathConfig,
    art_config: ArtConfig,
    port: int | None = None,
    seed: int | None = None,
) -> art.TrainableModel:
    if path_config.base_model != art_config.base_model:
        raise ValueError(f"Model name mismatch: {path_config.base_model} != {art_config.base_model}.")

    if port is None:
        port = find_free_port()
        logger.info(f"Found free port for ART server: {port}")

    art_config = art_config.initialize(output_dir=path_config.model_output_dir, port=port, seed=seed)
    logger.info(f"ART Model Config: {art_config.model_dump_json(indent=2)}")

    model = art.TrainableModel(
        name=path_config.run_name,
        project=path_config.project_name,
        base_model=path_config.base_model,
        _internal_config=art_config.internal_config,
    )

    backend = LocalBackend(path=path_config.art_path, in_process=True)
    await model.register(backend, _openai_client_config=art_config.openai_config)
    return model


def load_vllm_model(
    vllm_config: VllmConfig,
    port: int | None = None,
    seed: int | None = None,
) -> list[str]:
    if port is None:
        port = find_free_port()
        logger.info(f"Found free port for vLLM server: {port}")

    vllm_config = vllm_config.initialize(port=port, seed=seed)
    logger.info(f"vLLM Model Config: {vllm_config.model_dump_json(indent=2)}")

    engine_args = vllm_config.openai_config.get("engine_args", {})
    server_args = vllm_config.openai_config.get("server_args", {})

    full_args = [
        *[
            f"--{key.replace('_', '-')}{f'={item}' if item is not True else ''}"
            for args in [engine_args, server_args]
            for key, value in args.items()
            for item in (value if isinstance(value, list) else [value])
            if item is not None
        ],
    ]

    return full_args
