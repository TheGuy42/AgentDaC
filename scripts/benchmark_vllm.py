import logging
import argparse
import sys
import pathlib
import os

from art.dev import EngineArgs, ServerArgs
from vllm.benchmarks import serve as serve_benchmark

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.utils.logging import create_logger, setup_logging
from src.utils.env import prepare_environment
from src.configs.models.vllm import available_configs, CONFIGS


logger = create_logger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model to benchmark",
    )

    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="The port on which the vLLM server is running (e.g., '8200').",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="benchmark_results",
    )

    return parser.parse_known_args()


def main(args: argparse.Namespace, extra_args: list[str]) -> None:
    model_name: str = args.model
    port: int = args.port
    result_dir: str = args.result_dir

    if model_name not in available_configs():
        logger.error(f"Model '{model_name}' is not available. Available models are: {available_configs()}")
        sys.exit(1)

    config = CONFIGS[model_name]
    config = config.initialize(port)
    engine_args: EngineArgs = config.openai_config["engine_args"]  # type: ignore
    server_args: ServerArgs = config.openai_config["server_args"]  # type: ignore

    dummy_parser = argparse.ArgumentParser()
    serve_benchmark.add_cli_args(dummy_parser)

    dummy_parser.set_defaults(
        endpoint_type="vllm",
        host=server_args.get("host", "0.0.0.0"),
        port=server_args.get("port", port),
        result_dir=result_dir,
        metric_percentiles="25,50,75,99",
        seed=0,
    )

    DEFAULT_INPUT_LEN = 256
    DEFAULT_MODEL_LEN = 2048
    DEFAULT_CONCURRENCY = 128

    max_model_len = engine_args.get("max_model_len", DEFAULT_MODEL_LEN)
    if max_model_len is None:
        max_model_len = DEFAULT_MODEL_LEN

    max_concurrency = engine_args.get("max_num_seqs", DEFAULT_CONCURRENCY)
    if max_concurrency is None:
        max_concurrency = DEFAULT_CONCURRENCY

    random_output_len = (max_model_len - 2 * DEFAULT_INPUT_LEN) // 4

    dummy_parser.set_defaults(
        num_prompts=max_concurrency * 5,
        max_concurrency=max_concurrency,
        dataset_name="random",
        random_input_len=DEFAULT_INPUT_LEN,
        random_output_len=random_output_len,
        random_range_ratio=0.85,
    )

    bench_args = dummy_parser.parse_args(args=["--model", model_name] + extra_args)

    if openai_key := server_args.get("api_key", "default"):
        os.environ["OPENAI_API_KEY"] = openai_key

    serve_benchmark.main(bench_args)


if __name__ == "__main__":
    prepare_environment()
    setup_logging(logging.WARNING)
    args, extra_args = parse_args()
    main(args, extra_args)
