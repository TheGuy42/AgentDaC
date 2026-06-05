import numpy as np
from typing import Any
from collections import defaultdict

import agentlightning as agl
from agentlightning.verl.daemon import AgentModeDaemon
from src.utils.logging import create_logger


logger = create_logger(__name__)


class VerlDaemon(AgentModeDaemon):
    """
    Custom implementation of the `agentlightning.verl.daemon.AgentModeDaemon` to support additional custom metrics.
    Searches for `custom_metrics` in the triplet metadata of the last triplet of the rollout, and includes them in the training and test metrics.
    """

    def _find_custom_metrics(self, triplets: list[agl.Triplet]) -> dict[str, Any] | None:
        for triplet in reversed(triplets):
            # custom metrics are stored as metadata in the last triplet
            metadata = triplet.metadata or {}
            custom_metrics = metadata.get("custom_metrics")
            if custom_metrics is None:
                continue

            if not isinstance(custom_metrics, dict):
                logger.error(f"Found custom_metrics in triplet metadata, but it is not a dict: {custom_metrics}")
                continue

            return custom_metrics

        return None

    def _collect_custom_metrics(self):
        """
        Collect custom metrics from the completed rollouts.
        The custom metrics should be stored in the `metadata` of the triplet with the key 'custom_metrics.{metric}'.
        """
        metric_dict = defaultdict(list)
        for rollout in self._completed_rollouts_v0.values():
            instance_metrics = self._find_custom_metrics(rollout.triplets or [])
            if instance_metrics is None:
                logger.warning(f"No custom metrics found for rollout {rollout.rollout_id}")
            
            if instance_metrics is not None:
                for k, v in instance_metrics.items():
                    metric_dict[str(k)].append(float(v))

        mean_metrics = {str(k): float(np.mean(v)) for k, v in metric_dict.items()}
        return mean_metrics

    def get_train_data_batch(self, max_prompt_length, max_response_length, device, global_steps):
        data_proto, metrics = super().get_train_data_batch(
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            device=device,
            global_steps=global_steps,
        )

        custom_metrics = self._collect_custom_metrics()
        custom_metrics = {f"train-custom/{k}": v for k, v in custom_metrics.items()}
        metrics.update(custom_metrics)
        return data_proto, metrics

    def get_test_metrics(self):
        metrics = super().get_test_metrics()
        custom_metrics = self._collect_custom_metrics()
        custom_metrics = {f"val-custom/{k}": v for k, v in custom_metrics.items()}
        metrics.update(custom_metrics)
        return metrics
