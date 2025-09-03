# AgentDaC - Divide and Conquer Agents with Reinforcement Learning

AgentDaC is a framework for building AI agents that use a divide-and-conquer approach with reinforcement learning. It supports multiple agent types (JSON, Regex, Marker) and integrates with VLLM for model serving.

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

Alternatively, if you prefer pip:

```bash
pip install -e .
```

## Dependencies

Key dependencies include:
- `openpipe-art` for training infrastructure  
- `transformers` for model handling
- `openai` for API clients
- `vllm` for model serving
- `wandb` for experiment tracking
- `torch` for deep learning

Optional dependencies:
- `python-dotenv` for environment variable loading
- `flake8` for code linting

## Configuration

The framework uses configuration classes for:
- Model configurations (`src/configs/models/`)
- Prompt configurations (`src/configs/prompts/`)
- Training configurations (`src/configs/train_config.py`)
- Decomposition configurations (`src/configs/decomp_config.py`)

## Usage

See the experiments directory for examples:
- `experiments/math_json/` - Math problem solving with JSON agents
- `experiments/math_regex/` - Math problem solving with Regex agents

## API Keys

Create an `api_keys/` directory with the following files if needed:
- `WANDB_KEY.txt` - Weights & Biases API key
- `OPENPIPE_KEY.txt` - OpenPipe API key  
- `HF_KEY.txt` - Hugging Face token
- `OPENAI_KEY.txt` - OpenAI API key

Alternatively, set these as environment variables.

## Development

The project follows modern Python practices with:
- Type hints throughout
- Pydantic for configuration management
- Async/await for concurrent operations
- Comprehensive logging

For development, install additional tools:
```bash
pip install flake8 black mypy
```