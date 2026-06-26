"""custom-category metrics for the `main_ppo_sync` trainer.
Per-sample metrics are written to `extra_fields["custom_metrics"]`.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Any

import numpy as np
import ray
from omegaconf import OmegaConf
import transfer_queue as tq
import verl.trainer.main_ppo_sync as mps
from verl.utils import tensordict_utils as tu
from src.utils.logging import create_logger


logger = create_logger(__name__)


def mean_custom_metrics(extra_field_dicts: list, metrics: dict, prefix: str) -> None:
    """Mean each `custom_metrics` key over the given per-sample `extra_fields` dicts.

    Robust to non-uniform keys: each key is averaged over only the samples that carry it
    (that is the whole point of moving metrics out of `reward_extra_info`; see bug verl#6830).
    """
    per_key: dict[str, list[float]] = defaultdict(list)
    for extra_field in extra_field_dicts:
        if not isinstance(extra_field, dict):
            continue
        for key, value in (extra_field.get("custom_metrics") or {}).items():
            if value is not None:
                per_key[key].append(float(value))

    for key, values in per_key.items():
        if values:
            metrics[f"{prefix}/{key}/mean"] = float(np.mean(values))


class CustomPPOTrainer(mps.PPOTrainer):
    """`main_ppo_sync.PPOTrainer` with additive `train-custom/` + `val-custom/` metrics.
    The metrics are computed from the per-sample `extra_fields["custom_metrics"]` dicts.
    """

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        non_padding = np.array([not tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        extra_fields = data.get("extra_fields")
        if extra_fields is None:
            return
        non_padding_dicts = []
        for i, ef in enumerate(extra_fields.tolist()):
            if i < len(non_padding) and non_padding[i]:
                non_padding_dicts.append(ef)

        mean_custom_metrics(non_padding_dicts, metrics, prefix="train-custom")

    def _validate(self) -> dict[str, float]:
        # NOTE: Verbatim copy of verl `main_ppo_sync.PPOTrainer._validate` plus 
        # the `CUSTOM` lines. Added custom-metrics logging to the validation metrics. 
        
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}

        val_extra_fields: list = []  # NOTE: CUSTOM

        for batch_dict in self.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            self.async_rollout_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer
            batch = self.replay_buffer.sample(partition_id="val", global_steps=self.global_steps)

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                self.checkpoint_manager.sleep_replicas()
                batch = self._compute_reward_colocate(batch)
                self.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            base_offset = len(sample_scores)
            session_to_sample_idx.update({session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)})

            text_data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"])
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            sample_uids.extend(data.pop("uid").tolist())
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            sample_turns.extend(data.pop("num_turns").tolist())
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    reward_extra_info = extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            if extra_fields_list is not None:  # NOTE: CUSTOM
                val_extra_fields.extend(extra_fields_list.tolist())

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)

            # 5. cleanup transfer queue and replay buffer
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
            self.replay_buffer.remove(batch.partition_id, batch.keys)

        # logger to wandb
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key for key in dump_all_keys for parts in [key.rsplit("_", 2)]]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()}
                | {"uid": dump_all_keys},
                dump_path=val_data_dir,
            )

        metric_dict = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        mean_custom_metrics(val_extra_fields, metric_dict, prefix="val-custom")  # NOTE: CUSTOM
        return metric_dict


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
