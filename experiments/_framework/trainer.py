from __future__ import annotations

import random
from typing import Any

from src.agents.base import BaseAgent
from src.configs import DecompConfig
from src.inference import VerlClient
from src.trainer import RolloutStage, VerlTrainer

from experiments._framework.agents import build_agent


class ExperimentTrainer(VerlTrainer):
    """A `VerlTrainer` whose agent is selected at runtime from `custom_configs.agent`.

    Subclasses implement `format_prompt` and `score_trajectory` (the latter composing
    the task's `answer_reward` with `format_reward`/`behavior_reward`). Agent
    construction and the (optionally randomized) decomposition budget are shared here.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_key: str = str(self.config.custom_configs.agent)

    def build_decomp_config(self, stage: RolloutStage) -> DecompConfig:
        dc = self.decomp_config
        max_depth, max_tasks, max_rounds = dc.max_depth, dc.max_tasks, dc.max_rounds

        # Optionally randomize the decomposition budget during training.
        if stage == RolloutStage.TRAIN:
            if self.extra_config.get("randomize_decomp_depth", False):
                max_depth = random.randint(0, dc.max_depth)
            if self.extra_config.get("randomize_decomp_tasks", False):
                max_tasks = random.randint(0, dc.max_tasks)
            if self.extra_config.get("randomize_decomp_rounds", False):
                max_rounds = random.randint(0, dc.max_rounds)

        return DecompConfig(max_depth=max_depth, max_tasks=max_tasks, max_rounds=max_rounds)

    def create_agent(self, client: VerlClient, stage: RolloutStage) -> BaseAgent:
        return build_agent(self, client, stage, self.agent_key)
