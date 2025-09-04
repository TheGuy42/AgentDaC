import os
import dotenv
import huggingface_hub as hf
from src.utils.logging import create_logger


logger = create_logger(__name__)


def prepare_environment(dotenv_path: str | None = None):
    if dotenv_path is None:
        dotenv_path = dotenv.find_dotenv()

    logger.info(f"Using .env file at: {dotenv_path}")
    token_dict = dotenv.dotenv_values(dotenv_path=dotenv_path, verbose=True)

    if not token_dict.get("OPENAI_API_KEY"):
        openai_token = "default"
        token_dict["OPENAI_API_KEY"] = openai_token
        logger.info(f"OpenAI API key set to '{openai_token}'")

    if not token_dict.get("HF_TOKEN"):
        token_dict["HF_TOKEN"] = hf.get_token()

    token_dict = {k: v for k, v in token_dict.items() if v is not None}  # Filter out empty tokens

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
