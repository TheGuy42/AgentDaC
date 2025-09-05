# EPIC — Quickstart

## Prerequisites

- Python 3.12+
- CUDA GPU(s) for training/inference
- uv (Python package manager)

Install uv (Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

```bash
# from repo root
uv sync
```

This creates .venv and installs dependencies from `pyproject.toml`/`uv.lock`.

## Environment variables (.env)

Create a `.env` file in the repo root with the API keys used for data, inference and logging:

```bash
HF_TOKEN=...
OPENAI_API_KEY=...
WANDB_API_KEY=...
```

## (Optional) Start additional vLLM inference servers

Run one server per GPU with a unique port.

```bash
# GPU 0 on port 8200
python scripts/run_vllm_server.py --model <MODEL_ID> --port 8200 --gpus 1

# GPU 1 on port 8201 (second server)
python scripts/run_vllm_server.py --model <MODEL_ID> --port 8201 --gpus 2
```

Note: Use the same `<MODEL_ID>` you’ll train/eval with (see available models via `--help`).

## Run experiments

Pattern:

```bash
python experiments/<exp_name>/run.py --model <MODEL_ID> [--gpus 0] [--vllm_ports 8200 8201] [--project <NAME>] [--run <RESUME>] [--eval]
```

Examples:

```bash
# Train Math experiment on GPU 0 using two vLLM servers
python experiments/math/run.py --model <MODEL_ID> --gpus 0 --vllm_ports 8200 8201

# Evaluation-only (no training) of a clean model
python experiments/math/run.py --model <MODEL_ID> --eval
```

Tips:

- `--vllm_ports` lets the trainer route inference across your running vLLM servers.
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
