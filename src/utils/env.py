import os
from dotenv import load_dotenv
from pathlib import Path


def prepare_environment(tokens_folder: str = "./api_keys"):
    load_dotenv()

    folder_path = Path(tokens_folder)

    wandb_token = token_from_file(folder_path / "WANDB_KEY.txt")
    if wandb_token:
        os.environ["WANDB_API_KEY"] = wandb_token

    openpipe_token = token_from_file(folder_path / "OPENPIPE_KEY.txt")
    if openpipe_token:
        os.environ["OPENPIPE_API_KEY"] = openpipe_token

    hf_token = token_from_file(folder_path / "HF_KEY.txt")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"  # Set OMP_NUM_THREADS to 1 to avoid multithreading issues with vLLM
    os.environ["NCCL_CUMEM_ENABLE"] = "0"  # To avoid vLLM bug with NCCL
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # To avoid vLLM issues with multiprocessing
    os.environ["ART_SERVER_TIMEOUT"] = str(60 * 5)  # increase timeout for ART server creation
    return os.environ


def token_from_file(path: str | Path, do_raise: bool = False) -> str:
    """
    Read an API key from a file.

    Args:
        path (str | pathlib.Path): Path to the file containing the API key.

    Returns:
        str: The API key.

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
            print(f"API key file not found at {path}")
            return ""
