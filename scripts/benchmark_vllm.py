import logging
import argparse
import sys
import pathlib
import pprint

from art.dev import EngineArgs
from vllm.benchmarks import serve as serve_benchmark

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.utils.logging import create_logger, setup_logging
from src.configs import vllm_configs


logger = create_logger(__name__)


def parse_args():
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

    return parser.parse_args()


def main(args: argparse.Namespace):
    model_name: str = args.model
    port: int = args.port

    if model_name not in vllm_configs.available_configs():
        logger.error(f"Model '{model_name}' is not available. Available models are: {vllm_configs.available_configs()}")
        sys.exit(1)

    config = vllm_configs.CONFIGS[model_name]
    config = config.initialize(port)
    engine_args: EngineArgs = config.openai_config["engine_args"]  # type: ignore

    dummy_parser = argparse.ArgumentParser()
    serve_benchmark.add_cli_args(dummy_parser)

    dummy_parser.set_defaults(
        model=model_name,
        endpoint_type="vllm",
        host="0.0.0.0",
        port=port,
        logprobs=-1,
        results_dir="benchmark_results",
        metric_percentiles="25,50,75,99",
        seed=0,
    )

    random_input_len = 256
    max_model_len = engine_args.get("max_model_len", 4096)
    if max_model_len is None:
        max_model_len = 4096
    random_output_len = (max_model_len - 2 * random_input_len) // 4
    max_concurrency = engine_args.get("max_num_seqs", 64)

    dummy_parser.set_defaults(
        num_prompts=1024,
        max_concurrency=max_concurrency,
        dataset_name="random",
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        random_range_ratio=0.85,
    )

    bench_args = dummy_parser.parse_args(args=[])

    pprint.pprint("Benchmarking Args:")
    pprint.pprint(vars(bench_args), indent=4)

    serve_benchmark.main(bench_args)


if __name__ == "__main__":
    setup_logging(logging.WARNING)
    args = parse_args()
    main(args)
