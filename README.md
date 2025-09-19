# EPIC — Quickstart

## Setup

### Prerequisites

- Python 3.12+
- CUDA GPU(s) for training/inference
- uv (Python package manager)

Install uv (Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# from repo root
uv sync
```

This creates .venv and installs dependencies from `pyproject.toml`/`uv.lock`.

### Environment variables (.env)

Create a `.env` file in the repo root with the API keys used for data, inference and logging:

```bash
HF_TOKEN=...
OPENAI_API_KEY=...
WANDB_API_KEY=...
```

## Run experiment

Pattern:

```bash
python experiments/<exp_name>/run.py [--gpus 0] [--project <NAME>] [--run <RESUME>] [--config_dir <EXP_CONFIG_DIR>][--eval]
```

### (Optional) Additional vLLM inference servers

Run one server per GPU with a unique port.

```bash
# GPU 1 on port 8200
python scripts/run_vllm_server.py --config <MODEL_CONFIG> --port 8200 --gpus 1

# GPU 2 on port 8201 (second server)
python scripts/run_vllm_server.py --config <MODEL_CONFIG> --port 8201 --gpus 2
```

Note: Use the same model in `<MODEL_CONFIG>` you’ll train/eval with.

### Examples:

```bash
# Train MATH experiment on GPU 0 using two additional vLLM servers on GPUs 1 and 2
python experiments/math/run.py --gpus 0 --vllm_ports 8200 8201

# Evaluation-only (no training) of a clean model
python experiments/math/run.py --gpus 0 --eval
```

Tips:

- `--vllm_ports` lets the trainer route inference across additional vLLM servers.
- If omitted, default backend is used for inference.
- Available experiments live under `experiments/` (e.g., `experiments/math/run.py`).

## Notebooks

Interactive training/evaluation notebooks live in `notebooks/`:

- `notebooks/train.ipynb`
- `notebooks/evaluate.ipynb`

Open in Jupyter/VS Code and select the project kernel (created by `uv sync`).

## Troubleshooting

- Ensure the ports you pass in `--vllm_ports` match running servers.
- Set `--gpus` to local device IDs. `CUDA_VISIBLE_DEVICES` is handled internally.
- Use `--silent` on runners to reduce log verbosity.
- Run `scripts/benchmark_vllm.py` to appropriatly set-up vLLM inference settings.
