from __future__ import annotations

import asyncio
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from src.agents.registry import AgentContext, create_agent
from src.backends.vllm.config import VllmConfig
from src.configs import DecompConfig, PromptConfig, RolloutConfig
from src.inference import OAIClient
from src.running.rollout import RolloutError, RolloutTask
from src.running.stage import RolloutStage
from src.trajectory import Trajectory
from src.utils.io import save_object
from src.utils.logging import create_logger
from src.utils.trajectory_writer import TrajectoryWriter


logger = create_logger(__name__)


def aggregate_metrics(results: list[Trajectory | RolloutError]) -> dict[str, float]:
    """Mean of `reward` and of every trajectory metric, plus `exception_rate` and `n`."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def add(key: str, value: float) -> None:
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1

    for result in results:
        if isinstance(result, RolloutError):
            add("exception_rate", 1.0)
            continue

        add("exception_rate", 0.0)
        add("reward", result.reward)
        for key, value in result.metrics.items():
            add(key, value)

    metrics = {key: total / counts[key] for key, total in sums.items()}
    metrics["n"] = float(len(results))
    return metrics


class VllmRunner:
    """Runs scored rollouts against a live OpenAI-compatible server. Trains nothing."""

    def __init__(
        self,
        config: VllmConfig,
        task: RolloutTask,
        agent_name: str,
        prompt_config: PromptConfig,
        decomp_config: DecompConfig,
        rollout_config: RolloutConfig,
        extra_config: dict[str, Any] | None = None,
        writer: TrajectoryWriter | None = None,
        run_name: str | None = None,
    ) -> None:
        self.config = config
        self.task = task
        self.agent_name = agent_name
        self.prompt_config = prompt_config
        self.decomp_config = decomp_config
        self.rollout_config = rollout_config
        self.extra_config: dict[str, Any] = extra_config or {}
        self.writer = writer
        self.run_name = run_name

        self.client = OAIClient(
            model_name=config.server.model_name,
            base_url=config.server.base_url,
            api_key=config.server.api_key,
        )

        self._wandb_run = None
        self._semaphore = asyncio.Semaphore(config.inference.max_concurrency)

    @property
    def wandb_run(self):
        """Lazily started; `None` unless `wandb_project` is set."""
        if self._wandb_run is None and self.config.wandb_project:
            try:
                import wandb
            except ImportError:
                logger.warning("wandb_project is set but wandb is not installed; skipping.")
                self.config.wandb_project = None
                return None

            self._wandb_run = wandb.init(project=self.config.wandb_project, name=self.run_name, job_type="eval")
        return self._wandb_run

    def close(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def report(self, summary: dict[str, float], stage: RolloutStage, step: int = 0) -> None:
        """Log the summary, send it to wandb, and write it beside the trajectories."""
        logger.info(f"[{stage.value}] " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(summary.items())))

        if (run := self.wandb_run) is not None:
            run.log({f"{stage.value}/{k}": v for k, v in summary.items()})

        if self.writer and self.writer.enabled:
            path = self.writer.stage_dir(stage, step) / "summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "stage": stage.value,
                "model": self.config.server.model_name,
                **summary,
            }

            save_object(data, path, overwrite=True)
            logger.info(f"Wrote summary to {path}")

    def _agent_context(self, stage: RolloutStage) -> AgentContext:
        return AgentContext(
            agent_key=self.agent_name,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            extra_config=self.extra_config,
            tool_parser=self.config.agent.tool_parser,
            reasoning_parser=self.config.agent.reasoning_parser,
            tokenizer=self.config.agent.tokenizer or self.config.server.model_name,
        )

    def _chat_kwargs(self, stage: RolloutStage) -> dict[str, Any]:
        return self.rollout_config.get_kwargs(stage)

    async def rollout_one(
        self,
        sample: dict,
        stage: RolloutStage,
        rollout_id: str,
        step: int = 0,
    ) -> Trajectory | RolloutError:
        async with self._semaphore:
            try:
                agent = create_agent(client=self.client, ctx=self._agent_context(stage))
                chat_kw = self._chat_kwargs(stage)
                trajectory = await self.task.rollout(agent=agent, sample=sample, stage=stage, chat_kwargs=chat_kw)

            except RolloutError as e:
                logger.error("Rollout %s failed: %s", rollout_id, e, exc_info=True)
                if self.writer:
                    self.writer.write(e.trajectory, rollout_id=f"failed/{rollout_id}", stage=stage, step=step)
                return e

            if self.writer:
                self.writer.write(trajectory, rollout_id=rollout_id, stage=stage, step=step)

            return trajectory

    async def rollout(
        self,
        dataset: list[dict],
        stage: RolloutStage = RolloutStage.TEST,
        step: int = 0,
    ) -> list[Trajectory | RolloutError]:
        """Rollout every sample in for `group_size` times, returning a list of trajectories or exceptions."""

        tasks = []
        for index, sample in enumerate(dataset):
            for group_index in range(self.config.inference.group_size):
                rollout_id = f"step-{step}-sample-{index}-group-{group_index}"
                tasks.append(self.rollout_one(sample, stage, rollout_id=rollout_id, step=step))

        return await tqdm_asyncio.gather(*tasks, desc=f"inference-{stage.value}")

    async def evaluate(
        self,
        datasets: dict[RolloutStage, list[dict]],
        step: int = 0,
    ) -> dict[RolloutStage, list[Trajectory | RolloutError]]:
        """Roll out every stage and report each one as it finishes."""

        results: dict[RolloutStage, list[Trajectory | RolloutError]] = {}

        for stage, dataset in datasets.items():
            logger.info(f"Running {stage.value} inference over {len(dataset)} samples...")
            results[stage] = await self.rollout(dataset, stage=stage, step=step)
            self.report(aggregate_metrics(results[stage]), stage, step)

        return results
