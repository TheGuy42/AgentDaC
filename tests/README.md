# Tests

```
tests/
├── conftest.py              # shared fixtures + CLI options for live tests
├── unit/                    # pure, fast, no server/GPU
│   ├── factories.py         # in-memory Response/Trajectory/Rollout builders
│   ├── test_convert.py      # convert_trajectory -> spans -> VerlAdapter triplets
│   ├── test_dicts.py        # src.utils.dicts get/set helpers
│   ├── test_rollout_config.py
│   ├── test_trajectory.py   # message conversion / for_logging
│   └── test_chat_kwargs.py  # AglTrainer.chat_kwargs precedence + flag injection
└── integration/             # LIVE: need a running vLLM/OpenAI-compatible server
    ├── agent_specs.py        # per-agent configs + shared assertions
    └── test_agent_smoke.py   # parametrized over every agent (replaces scripts/smoke_*.py)
```

## Install (one-time)

The suite uses `pytest` (+ `pytest-asyncio` for the live tests). With uv:

```bash
uv sync --group dev
```

## Run unit tests (no server, no GPU)

```bash
pytest tests/unit
```

## Run live agent smoke tests

Skipped automatically unless `--model` is given:

```bash
pytest tests/integration \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --base-url http://0.0.0.0:8000/v1 \
  --api-key EMPTY
```

Options: `--base-url`, `--model`, `--api-key`, `--max-completion-tokens`.
