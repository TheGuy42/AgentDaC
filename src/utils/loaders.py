import art
from art.local import LocalBackend
import src.configs.models.art as art_configs
import src.configs.models.vllm as vllm_configs
from src.configs import PathConfig, VllmConfig, ArtConfig
from src.utils.logging import create_logger


logger = create_logger(__name__)


async def load_art_model(
    path_config: PathConfig,
    art_config: ArtConfig | None = None,
    seed: int | None = None,
    print_full: bool = False,
) -> art.TrainableModel:
    if art_config is None:
        if path_config.base_model not in art_configs.CONFIGS:
            raise ValueError(
                f"No configuration found for model: {path_config.base_model}. "
                f"Available configs: {art_configs.available_configs()}"
            )

        logger.info(f"Loading default config for {path_config.base_model}...")
        art_config = art_configs.CONFIGS[path_config.base_model]

    if path_config.base_model != art_config.base_model:
        raise ValueError(f"Model name mismatch: {path_config.base_model} != {art_config.base_model}.")        

    if not print_full:
        print("Art Config:")
        print(art_config.model_dump_json(indent=2))

    art_config = art_config.initialize(output_dir=path_config.model_output_dir, seed=seed)

    if print_full:
        print("Full Art Config:")
        print(art_config.model_dump_json(indent=2))

    model = art.TrainableModel(
        name=path_config.run_name,
        project=path_config.project_name,
        base_model=path_config.base_model,
        _internal_config=art_config.internal_config,
    )

    backend = LocalBackend(path=path_config.art_path)
    await model.register(backend, _openai_client_config=art_config.openai_config)
    return model


def load_vllm_model(
    model_name: str,
    port: int = 8200,
    seed: int | None = None,
    vllm_config: VllmConfig | None = None,
    print_full: bool = False,
) -> list[str]:
    if vllm_config is None:
        if model_name not in vllm_configs.CONFIGS:
            raise ValueError(
                f"No configuration found for model: {model_name}. Available configs: {vllm_configs.available_configs()}"
            )

        logger.info(f"Loading default config for {model_name}...")
        vllm_config = vllm_configs.CONFIGS[model_name]

    if vllm_config.base_model != model_name:
        raise ValueError(f"Model name mismatch: {vllm_config.base_model} != {model_name}.")

    if not print_full:
        print("vLLM Config:")
        print(vllm_config.model_dump_json(indent=2))

    vllm_config = vllm_config.initialize(port=port, seed=seed)

    if print_full:
        print("Full vLLM Config:")
        print(vllm_config.model_dump_json(indent=2))

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
