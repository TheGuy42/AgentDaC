import os
from dotenv import load_dotenv
from pathlib import Path
import huggingface_hub as hf
from src.utils.logging import create_logger


logger = create_logger(__name__)


def prepare_environment(tokens_folder: str = "./api_keys"):
    load_dotenv()

    folder_path = Path(tokens_folder)

    token_dict = {
        "WANDB_API_KEY": token_from_file(folder_path / "WANDB_KEY.txt"),
        "OPENPIPE_API_KEY": token_from_file(folder_path / "OPENPIPE_KEY.txt"),
        "HF_TOKEN": token_from_file(folder_path / "HF_KEY.txt") or hf.get_token(),
        "OPENAI_API_KEY": token_from_file(folder_path / "OPENAI_KEY.txt"),
    }

    token_dict = {k: v for k, v in token_dict.items() if v}  # Filter out empty tokens

    if "OPENAI_API_KEY" not in token_dict:
        token_dict["OPENAI_API_KEY"] = "default"
        logger.info("OpenAI API key set to 'default'")

    os.environ.update(token_dict)
    logger.info(f"Setting API tokens: {list(token_dict.keys())}")

    flag_dict = {
        # "TORCHINDUCTOR_MAX_AUTOTUNE": "1",  # Set to 1 to avoid multithreading issues with vLLM
        # "OMP_NUM_THREADS": "1",  # Set OMP_NUM_THREADS to 1 to avoid multithreading issues with vLLM
        "NCCL_CUMEM_ENABLE": "0",  # To avoid vLLM bug with NCCL
        "VLLM_USE_V1": "0",  # NOTE: Currently ART uses vLLM v0, see art.unsloth.state
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",  # To avoid vLLM issues with multiprocessing
        "ART_SERVER_TIMEOUT": str(60 * 5),  # increase timeout for ART
        "WEAVE_DISABLED": "1",
        "WEAVE_DISABLE_TRACING": "1",
    }
    
    os.environ.update(flag_dict)
    logger.info(f"Setting additional variables: {flag_dict}")


def token_from_file(path: str | Path, do_raise: bool = False) -> str | None:
    """
    Read an API key from a file.

    Args:
        path (str | pathlib.Path): Path to the file containing the API key.
        do_raise (bool): If True, raises FileNotFoundError if the file does not exist.
            If False, returns None and logs a message.

    Returns:
        (str | None): The API key if found, None otherwise.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    if isinstance(path, str):
        path = Path(path)

    key_file = Path(path).resolve()

    if key_file.exists():
        with key_file.open("r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        if do_raise:
            raise FileNotFoundError("API key file not found")
        else:
            logger.info(f"API key file not found at {path}")
            return None
