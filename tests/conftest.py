"""Shared pytest configuration.

Unit tests under ``tests/unit`` are pure/in-memory (no server, no GPU).

Integration tests under ``tests/integration`` are *live* and need a running
OpenAI-compatible (vLLM) server. They are skipped unless ``--model`` is passed, e.g.::

    pytest tests/integration --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://0.0.0.0:8000/v1
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("live-server", "Options for live (integration) agent tests")
    group.addoption("--base-url", action="store", default="http://0.0.0.0:8000/v1",
                    help="Base URL of the OpenAI-compatible inference server.")
    group.addoption("--model", action="store", default=None,
                    help="Served model name. Integration tests are skipped when omitted.")
    group.addoption("--api-key", action="store", default="EMPTY",
                    help="API key for the inference server.")
    group.addoption("--max-completion-tokens", action="store", type=int, default=256,
                    help="Cap on completion tokens for live agent calls.")


@pytest.fixture(scope="session")
def base_url(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--base-url")


@pytest.fixture(scope="session")
def api_key(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--api-key")


@pytest.fixture(scope="session")
def max_completion_tokens(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--max-completion-tokens")


@pytest.fixture(scope="session")
def model(request: pytest.FixtureRequest) -> str:
    """Served model name; skips the test when no --model was provided."""
    value = request.config.getoption("--model")
    if not value:
        pytest.skip("live server tests require --model (and a reachable --base-url)")
    return value
