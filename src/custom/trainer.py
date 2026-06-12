from __future__ import annotations
import random
from copy import deepcopy

import numpy as np
import torch
from src.utils.logging import create_logger

from agentlightning.verl.trainer import (
    AgentLightningTrainer,
    DataProto,
    AdvantageEstimator,  # type: ignore[import]
    _timer,
    compute_data_metrics,
    pad_dataproto_to_divisor,
    unpad_dataproto,
    agg_loss,
    compute_throughout_metrics,
    compute_timing_metrics,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    reduce_metrics,
)


logger = create_logger(__name__)


class VerlTrainer(AgentLightningTrainer):
    """
    Custom implementation of the `agentlightning.verl.trainer.AgentLightningTrainer`.

    This is the exact same implementation as the original, except with the following bugfixes:
    - https://github.com/microsoft/agent-lightning/issues/492
    - https://github.com/microsoft/agent-lightning/issues/50

    Additionally, it attaches a `custom_meta` dict to every sample handed to the daemon
    (see `_attach_custom_meta`), carrying dynamic metadata such as the current training step.
    """

    def _attach_custom_meta(self, non_tensor_batch: dict) -> None:
        """
        Stamp a `custom_meta` dict column onto every sample of the batch (in place).

        It holds dynamic metadata that is only known at training time — currently the
        `step_number` of the batch; extend the dict below as needed.

        The daemon copies every column of this dict into the task delivered to the runners,
        so it arrives in `rollout_async` as `task["custom_meta"]` (e.g. for trajectory
        logging). Related: https://github.com/microsoft/agent-lightning/issues/401
        """
        num_samples = len(next(iter(non_tensor_batch.values())))
        non_tensor_batch["custom_meta"] = np.array(
            [{"step_number": self.global_steps} for _ in range(num_samples)], dtype=object
        )

    # NOTE: exactly the same implementation as the upstream `_validate`, except that
    # the samples are stamped with `custom_meta` (see `_attach_custom_meta`).
    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."

        test_data = next(iter(self.val_dataloader))
        test_batch = DataProto.from_single_dict(test_data)

        self.async_rollout_manager.wake_up()
        self._attach_custom_meta(test_batch.non_tensor_batch)
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.async_rollout_manager.server_addresses,
            is_train=False,
        )
        self.agent_mode_daemon.run_until_all_finished()
        test_metrics = self.agent_mode_daemon.get_test_metrics()
        self.agent_mode_daemon.clear_data_and_server()
        self.async_rollout_manager.sleep()
        return test_metrics

    # NOTE: exactly the same implementation, except with the following bugfix:
    # https://github.com/microsoft/agent-lightning/issues/492
    def _train_step(self, batch_dict: dict) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        with _timer("step", timing_raw):
            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            with _timer("gen", timing_raw):
                self.async_rollout_manager.wake_up()
                self._attach_custom_meta(gen_batch.non_tensor_batch)
                self.agent_mode_daemon.set_up_data_and_server(gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses)
                self.agent_mode_daemon.run_until_all_finished()
                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=(
                        self.config.agentlightning.trace_aggregator.trajectory_max_prompt_length
                        if self.config.agentlightning.trace_aggregator.level.startswith("trajectory")
                        else self.config.data.max_prompt_length
                    ),
                    max_response_length=(
                        self.config.agentlightning.trace_aggregator.trajectory_max_response_length
                        if self.config.agentlightning.trace_aggregator.level.startswith("trajectory")
                        else self.config.data.max_response_length
                    ),
                    device=gen_batch.batch["fake_ids"].device,
                    global_steps=self.global_steps,
                )
                metrics.update(agent_metrics)
                self.agent_mode_daemon.clear_data_and_server()
                self.async_rollout_manager.sleep()

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with _timer("gen_max", timing_raw):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]

            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_extra_infos_dict = {}

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self._compute_reference_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)

            with _timer("adv", timing_raw):
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # Calculate the metrics before processing. Refer to the comments of function `compute_data_metrics` for details.
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, suffix="_before_processing"))

            # after advantages are assinged, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["training/n_triplets_prompt_too_long"] = batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            batch = batch[keep_indices]
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)
            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["training/n_triplets_dropped_remainder"] = n_transition - n_remained_transition

            # NOTE: this is a patch - If all samples are too long then we will have an empty batch, which will cause errors.
            if len(batch) == 0:
                raise RuntimeError(
                    "All samples in the batch are dropped due to long prompts. Skipping training step. "
                    "Consider increasing `config.data.max_prompt_length` or "
                    "`config.agentlightning.trace_aggregator.trajectory_max_prompt_length` (if using config.agentlightning.trace_aggregator.level=trajectory)"
                )

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with _timer("dump_rollout_generations", timing_raw):
                    inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    self._dump_generations(
                        inputs=inputs,
                        outputs=outputs,
                        gts=[None for _ in range(len(inputs))],  # NOTE: this is a patch, required argument that was missing.
                        scores=scores,
                        reward_extra_infos_dict=reward_extra_infos_dict,
                        dump_path=rollout_data_dir,
                    )

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, suffix="_after_processing"))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        return metrics
