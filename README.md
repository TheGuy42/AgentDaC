# AgentDaC

AgentDaC is a research framework for training and evaluating recursive LLM agents. An
agent can answer directly, reason over several turns, delegate independent subproblems,
or maintain a persistent conversation with a sub-agent.

The core agent and task code is independent of the training framework. Backend adapters
connect the same rollout logic to VERL, ART, or an OpenAI-compatible vLLM server.

## Architecture

The main abstractions are:

- `RolloutTask`: formats one dataset sample and scores the resulting trajectory.
- `BaseAgent`: implements an agent protocol and its delegation behavior.
- `InferenceClient`: sends chat turns to a backend without exposing backend details to
  the agent.
- `Trajectory`: records messages, model responses, nested histories, rewards, metrics,
  metadata, logs, and errors.
- `TaskDataset`: loads task data and prepares deterministic train, validation, and test
  splits.
- Backend adapters: assemble configurations, run rollouts, and optionally train the model.

```text
experiment entry point
        |
        v
dataset -> task -> agent -> inference client -> model backend
                   |                              |
                   +--------- trajectory <-------+
                                  |
                                  v
                         scoring and training
```

Repository layout:

```text
src/
  agents/          agent protocols and parsers
  inference/       backend-independent inference interface
  running/         task, dataset, and rollout contracts
  backends/        VERL, ART, and vLLM adapters
  configs/         typed shared configuration
  trajectory.py    canonical rollout representation
experiments/
  _framework/      shared experiment runner and backend dispatch
  <task>/           task-specific data, prompting, scoring, and configs
config_files/
  prompts/         reusable system prompts
  templates/       chat-template overrides
```

## Installation

AgentDaC uses Python 3.12 and `uv`. Choose exactly one backend extra; the backend extras
are intentionally mutually exclusive. The `experiments` extra adds the dependencies used
by the checked-in tasks.

```bash
# VERL training
uv sync --extra verl --extra experiments

# ART training
uv sync --extra art --extra experiments

# Inference/evaluation against a running vLLM server
uv sync --extra vllm --extra experiments
```

Run commands from the repository root. Checked-in prompt and chat-template paths are
repository-relative.

Depending on the selected backend and its logging configuration, a `.env` file may
contain:

```dotenv
HF_TOKEN=...
WANDB_API_KEY=...
OPENAI_API_KEY=...
```

Not every experiment/agent directory provides configuration for every backend. Select a
combination that contains the backend-specific files described below.

## Backends

### VERL

The VERL backend performs reinforcement-learning rollouts and training. It adapts a
multi-turn `Trajectory` into VERL's token-level agent-loop output while retaining the
sampled response token IDs and masking controller, user, tool, and returned sub-agent
tokens from the policy loss.

Required backend files:

```text
verl_config.yaml
verl_rollout_config.yaml
```

Example:

```bash
uv run python experiments/math/run.py \
  --backend verl \
  --agent marker \
  --gpus 0 \
  --test_run
```

### ART

The ART backend runs local ART rollout groups and training. It supports training a whole
trajectory or splitting a multi-turn trajectory into episodes, depending on the ART
configuration.

Required backend files:

```text
art_config.yaml
art_rollout_config.yaml
```

Example:

```bash
uv run python experiments/memorization/run.py \
  --backend art \
  --agent dummy \
  --gpus 0 \
  --test_run
```

### vLLM

The vLLM backend is inference-only. It connects to an already running
OpenAI-compatible server, performs scored rollouts, and optionally logs trajectories and
metrics. It does not launch the server or train a model.

Required backend files:

```text
vllm_config.yaml
vllm_rollout_config.yaml
```

Example:

```bash
uv run python experiments/chess/run.py \
  --backend vllm \
  --agent tool_submit \
  --gpus 0 \
  --test_run
```

The server URL, served model name, parser settings, requested splits, group size, and
concurrency are configured in `vllm_config.yaml`.

If `--backend` is omitted, AgentDaC selects the first supported backend it detects in the
active environment. Passing it explicitly is clearer when more than one backend package
is installed.

## Experiments and agents

Checked-in experiments include mathematical reasoning, Easy2Hard, BBEH, Saturn,
synthetic memorization, and chess move selection. Each experiment's `run.py` is the source
of truth for its supported agent protocols.

The available protocols include direct response, marker-delimited actions, guided JSON
or regex actions, persistent sub-agents, and native tool-calling variants. Agent builders
are registered in `src/agents/registry.py`.

Common command-line options:

- `--agent`: agent protocol; required.
- `--backend`: `verl`, `art`, or `vllm`; optional auto-detection when omitted.
- `--gpus`: visible local GPU IDs.
- `--config_dir`: override the default `experiments/<task>/configs/<agent>` directory.
- `--project`: logging project name.
- `--run`: run name.
- `--traj_dir`: enable full trajectory JSON logging under this directory.
- `--seed`: experiment seed.
- `--silent`: reduce logging.
- `--test_run`: apply small, backend-specific smoke-run settings.

Use an experiment entry point for its exact choices:

```bash
uv run python experiments/chess/run.py --help
```

## Configuration

Every configured experiment/agent combination starts with shared files:

```text
data_config.yaml       dataset sizes, seed, and task-specific loader parameters
prompt_config.yaml     root, intermediate, and leaf system prompts
decomp_config.yaml     depth, delegation, and round budgets
extra_config.yaml      optional experiment or agent extension settings
```

It then adds the two files required by its backend. Tasks may add their own typed files;
chess, for example, adds `chess_config.yaml` and `engine_config.yaml`.

Typed structural configuration rejects unknown fields. Deliberate extension surfaces
remain open: dataset-specific fields in `data_config.yaml`, raw `extra_config.yaml`, raw
VERL overrides, rollout argument dictionaries, and backend-native ART model settings.

Generation arguments are layered as:

```text
kwargs < train_kwargs | val_kwargs | test_kwargs
```

The stage-specific dictionary overrides the shared dictionary.

## Trajectory output

Passing `--traj_dir <directory>` writes readable JSON trajectories beneath a
backend/project/run directory:

```text
<traj_dir>/<backend>/<project>/<run>/step_<step>/<train|val|test>/<rollout_id>.json
```

Each record includes the conversation, tool schemas, reward, custom metrics, metadata,
logs, errors, and any retained nested histories.

## Extending AgentDaC

To add a task:

1. Implement `TaskDataset` and `RolloutTask` subclasses.
2. Add task formatting and reward logic.
3. Create an experiment `run.py` with its supported agents.
4. Add shared and backend-specific configuration files.

To add an agent protocol:

1. Implement `BaseAgent.chat()`, `parse_answer()`, and `error_kinds()`.
2. Add any parser, action, or tool-schema support it needs.
3. Register its builder in `src/agents/registry.py`.

The backend-independent contracts should remain free of trainer-specific types; backend
conversion belongs under `src/backends/`.
