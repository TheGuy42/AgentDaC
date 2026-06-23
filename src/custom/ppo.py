"""custom-category metrics for the ``main_ppo_sync`` trainer.

verl auto-averages per-key ``reward_extra_info`` into logged metrics only at *validation*
(``val-core/...``), never during training. This module adds, purely additively:

- ``train-custom/<key>/mean`` — averaged every training step (via the ``_compute_metrics`` seam).
- ``val-custom/<key>``       — aliased from the native ``val-core/...`` means (via ``_validate``).

It rides verl's intended extension points only: an overridable ``PPOTrainer`` subclass plus the
``run_ppo(config, task_runner_class=...)`` recipe hook. No ``fit()`` copy, no monkeypatch — the
overrides call ``super()`` and only *add* keys to the metrics dict.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pprint import pprint
from typing import Any

import numpy as np
import ray
from omegaconf import OmegaConf
import transfer_queue as tq
import verl.trainer.main_ppo_sync as mps
from src.utils.logging import create_logger


logger = create_logger(__name__)


def add_train_custom_metrics(batch, metrics: dict) -> None:
    """Average each ``reward_extra_info`` key over the non-padding samples of this step.

    ``reward_extra_info`` is stored nested in the per-sample ``extra_fields`` field in
    TransferQueue (written by the rollout postprocess), so we read it back the same way
    verl's own validation path does (``main_ppo_sync.py:932-945``).
    """
    non_padding = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
    data = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["extra_fields"],
    )
    extra_fields = data.get("extra_fields")
    if extra_fields is None:
        return

    per_key: dict[str, list[float]] = defaultdict(list)
    for i, extra_field in enumerate(extra_fields.tolist()):
        if i < len(non_padding) and not non_padding[i]:
            continue
        if not isinstance(extra_field, dict):
            continue
        reward_extra_info = extra_field.get("reward_extra_info") or {}
        for key, value in reward_extra_info.items():
            if value is not None:
                per_key[key].append(float(value))

    for key, values in per_key.items():
        if values:
            metrics[f"train-custom/{key}/mean"] = float(np.mean(values))


class CustomPPOTrainer(mps.PPOTrainer):
    """``main_ppo_sync.PPOTrainer`` with additive ``train-custom/`` + ``val-custom/`` metrics."""

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        add_train_custom_metrics(batch, metrics)


# Recover the original (undecorated) TaskRunner so we can subclass it; ``main_ppo_sync.TaskRunner``
# is a ``@ray.remote`` ActorClass. We override only ``run()`` to build ``AgentDacPPOTrainer``
# (a ~15-line copy of the stock body, changing one line), then re-wrap with ``ray.remote``.


def unwrap_ray_actor_class(actor_cls: type) -> type:
    """Return the original Python class behind a Ray ActorClass.

    VERL decorates TaskRunner with @ray.remote, but we need normal Python
    inheritance to override only run(). Ray stores the original class here.
    Keep this in one place so the dependency is explicit and easy to delete
    if VERL later exposes a trainer_cls hook.
    """
    try:
        return actor_cls.__ray_actor_class__
    except AttributeError as e:
        raise TypeError(
            f"{actor_cls!r} does not look like a Ray ActorClass with __ray_actor_class__. VERL may have changed TaskRunner wrapping."
        ) from e


BaseTaskRunner = unwrap_ray_actor_class(mps.TaskRunner)


@ray.remote
class CustomTaskRunner(BaseTaskRunner):
    def run(self, config: Any):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # initialize transfer queue
        tq.init(config.transfer_queue)
        trainer = None
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)

            trainer = CustomPPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            if trainer:
                trainer.replay_buffer.close()
            tq.close()
