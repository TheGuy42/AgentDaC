from src.dac_agent import AgentNode
from src.dac_agent_single import SingleAgentNode
from src.dac_agent_all_traj import AllTrajSingleTaskAgentNode
from src.dac_agent_single_frozen import SingleAgentNodeFrozen
from src.trainer import Trainer
from src.configs.markers import Markers
from src.utils.text import extract_answer, extract_between
from src.dac_agent import ChatMessage, PromptConfig, StopCriteria
from src.vllm_client import VllmRouter
from src.models import PathConfig
from src.utils.replay import GeneralReplayBuffer, RewardBasedDoubleQuantileReplayBuffer

from experiments.general_rewards import format_reward, behavior_reward
from experiments.easy2hard_guy.rewards import answer_reward, verify
from experiments.easy2hard_guy.format import format_prompt

import art
from art.utils import output_dirs
import random
import os


class Easy2HardTrainer(Trainer):
    include_histories: bool = False
    def create_agent(self) -> SingleAgentNode:
        client = self.vllm_router.next()
        return SingleAgentNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            include_histories=self.include_histories,
        )

    async def forward_step(self, sample: dict, **kwargs) -> art.Trajectory:
        content = format_prompt(sample)
        agent = self.create_agent()
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    async def rollout_step(self, sample: dict, **kwargs) -> art.Trajectory:
        # Perform a forward step to get the trajectory
        trajectory = await self.forward_step(sample, **kwargs)

        ans_message = ChatMessage.model_validate(trajectory.messages()[-1], from_attributes=True)

        train_step = await self.model.get_step() # type: TrainableModel
        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = answer_reward(sample, ans_message)# if train_step > 7 else 0.0
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)# * 0.1
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
                "is_correct": int(verify(answer, agent_answer)),
                "gave_answer": int(num_answers > 0),
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "problem": problem,
                "answer": answer,
                "agent_answer": agent_answer,
                "item_difficulty": sample["item_difficulty"],
                "index": sample["index"],
            }
        )

        return trajectory

class Easy2HardTrainerVariableDepth(Trainer):
    include_histories: bool = False
    def create_agent_v(self, stop_condition:StopCriteria) -> SingleAgentNode:
        client = self.vllm_router.next()
        return SingleAgentNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=stop_condition,#self.stop_criteria,
            include_histories=self.include_histories,
        )

    async def forward_step_v(self, sample: dict, stop_condition:StopCriteria, **kwargs) -> art.Trajectory:
        content = format_prompt(sample)
        agent = self.create_agent_v(stop_condition)
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    async def rollout_step_v(self, sample: dict, stop_condition:StopCriteria, **kwargs) -> art.Trajectory:
        # Perform a forward step to get the trajectory
        trajectory = await self.forward_step_v(sample, stop_condition, **kwargs)

        ans_message = ChatMessage.model_validate(trajectory.messages()[-1], from_attributes=True)

        train_step = await self.model.get_step() # type: TrainableModel
        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = answer_reward(sample, ans_message)# if train_step > 7 else 0.0
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)# * 0.1
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
                "is_correct": int(verify(answer, agent_answer)),
                "gave_answer": int(num_answers > 0),
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "problem": problem,
                "answer": answer,
                "agent_answer": agent_answer,
                "item_difficulty": sample["item_difficulty"],
                "index": sample["index"],
                "stop_depth": stop_condition.max_depth,
            }
        )

        return trajectory
    
    async def rollout(
        self,
        dataset: list[dict],
        group_size: int,
        max_exceptions: int | float = 0,
        pbar_desc: str = "rollout",
        **kwargs,
    ) -> list[art.TrajectoryGroup]:
        """
        Performs a grouped rollout for the given dataset, generating a trajectory for each sample.
        The dataset is split into groups of size `group_size` defined in the training configuration.

        Args:
            dataset (list[dict]): List of samples to process.
            group_size (int): Number of rollouts for each sample.
            max_exceptions (int | float): Maximum number of failed trajectories to allow before failing.
                If float, then it is treated as a fraction of the dataset size.
            pbar_desc (str): Description for the progress bar.
            **kwargs: Additional keyword arguments for the rollout step.

        Returns:
            list[art.TrajectoryGroup]: List of trajectory groups generated by the model for the dataset.
        """
        if not kwargs:
            kwargs = self.default_kwargs

        groups = []
        for sample in dataset:
            assert self.stop_criteria.max_depth is not None
            sample_stop_criteria = self.stop_criteria.clone()
            if "train" in pbar_desc: #TODO: better way to identify training
                depth = random.choices(range(self.stop_criteria.max_depth+1), weights=[0.3 / self.stop_criteria.max_depth]*self.stop_criteria.max_depth+[0.7])
                sample_stop_criteria.max_depth = depth[0]
            group = art.TrajectoryGroup([self.rollout_step_v(sample, sample_stop_criteria, **kwargs) for _ in range(group_size)])
            groups.append(group)

        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc=pbar_desc,
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
        )

        return trajectory_groups

class Easy2HardTrainerVariableDepthReplay(Trainer):
    include_histories: bool = False
    def __init__(
        self,
        model: art.Model,
        vllm_router: VllmRouter,
        path_config: PathConfig,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
        default_kwargs: dict | None = None,
        buffer_ratio: float = 0.3,
    ):
        super().__init__(model, vllm_router, path_config, prompt_config, stop_criteria, default_kwargs)
        train_logs_path = output_dirs.get_trajectories_split_dir(self.path_config.model_output_dir, "train")
        os.makedirs(train_logs_path, exist_ok=True)
        self.replay_buffer: GeneralReplayBuffer = RewardBasedDoubleQuantileReplayBuffer(
            directory=train_logs_path,
            grouping_keys=["index", "stop_depth"],
            upper_quantile=0.8,
            buffer_size=70,  # Only keep the last N epoch files
        )
        self.buffer_ratio:float = buffer_ratio

    def create_agent_v(self, stop_condition:StopCriteria) -> SingleAgentNode:
        client = self.vllm_router.next()
        return SingleAgentNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=stop_condition,
            include_histories=self.include_histories,
        )

    async def forward_step_v(self, sample: dict, stop_condition:StopCriteria, **kwargs) -> art.Trajectory:
        content = format_prompt(sample)
        agent = self.create_agent_v(stop_condition)
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    async def rollout_step_v(self, sample: dict, stop_condition:StopCriteria, **kwargs) -> art.Trajectory:
        # Perform a forward step to get the trajectory
        trajectory = await self.forward_step_v(sample, stop_condition, **kwargs)

        ans_message = ChatMessage.model_validate(trajectory.messages()[-1], from_attributes=True)

        train_step = await self.model.get_step() # type: TrainableModel
        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = answer_reward(sample, ans_message)# if train_step > 7 else 0.0
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)# * 0.1
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
                "is_correct": int(verify(answer, agent_answer)),
                "gave_answer": int(num_answers > 0),
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "problem": problem,
                "answer": answer,
                "agent_answer": agent_answer,
                "item_difficulty": sample["item_difficulty"],
                "index": sample["index"],
                "stop_depth": stop_condition.max_depth,
                "replay": False,
            }
        )

        return trajectory
    
    async def rollout(
        self,
        dataset: list[dict],
        group_size: int,
        max_exceptions: int | float = 0,
        pbar_desc: str = "rollout",
        **kwargs,
    ) -> list[art.TrajectoryGroup]:
        """
        Performs a grouped rollout for the given dataset, generating a trajectory for each sample.
        The dataset is split into groups of size `group_size` defined in the training configuration.

        Args:
            dataset (list[dict]): List of samples to process.
            group_size (int): Number of rollouts for each sample.
            max_exceptions (int | float): Maximum number of failed trajectories to allow before failing.
                If float, then it is treated as a fraction of the dataset size.
            pbar_desc (str): Description for the progress bar.
            **kwargs: Additional keyword arguments for the rollout step.

        Returns:
            list[art.TrajectoryGroup]: List of trajectory groups generated by the model for the dataset.
        """
        if not kwargs:
            kwargs = self.default_kwargs

        groups = []
        for sample in dataset:
            assert self.stop_criteria.max_depth is not None
            sample_stop_criteria = self.stop_criteria.clone()
            if "train" in pbar_desc: #TODO: better way to identify training
                depth = random.choices(range(self.stop_criteria.max_depth+1), weights=[0.3 / self.stop_criteria.max_depth]*self.stop_criteria.max_depth+[0.7])
                sample_stop_criteria.max_depth = depth[0]
            group = art.TrajectoryGroup([self.rollout_step_v(sample, sample_stop_criteria, **kwargs) for _ in range(group_size)])
            groups.append(group)

        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc=pbar_desc,
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
        )
        import logging
        # Load replay trajectories and add to the output
        if "train" in pbar_desc: #TODO: better way to identify training
            new_files = self.replay_buffer.update_trajectories()
            logging.info(f"Loaded {new_files} new trajectories into the replay buffer, total size now {self.replay_buffer.num_files_loaded}")

            for traj_group in trajectory_groups:
                group_key = tuple(traj_group.trajectories[0].metadata[k] for k in self.replay_buffer.grouping_keys)
                n = max(1, int(len(traj_group.trajectories) * self.buffer_ratio))
                replay_buffer = self.replay_buffer.sample_group(group_key=group_key, n=n)
                for traj in replay_buffer: # mark as replay
                    traj.metadata["replay"] = True
                traj_group.trajectories.extend(replay_buffer) # add replay to the group

        return trajectory_groups


class Easy2HardTrainerFrozen(Trainer):
    include_histories: bool = False
    def create_agent(self) -> SingleAgentNodeFrozen:
        client = self.vllm_router.next()
        return SingleAgentNodeFrozen(
            model_name=self.model.get_inference_name(),
            base_model_name=client.base_model,
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            include_histories=self.include_histories,
        )

    async def forward_step(self, sample: dict, **kwargs) -> art.Trajectory:
        content = format_prompt(sample)
        agent = self.create_agent()
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)
        return trajectory

    async def rollout_step(self, sample: dict, **kwargs) -> art.Trajectory:
        # Perform a forward step to get the trajectory
        trajectory = await self.forward_step(sample, **kwargs)

        ans_message = ChatMessage.model_validate(trajectory.messages()[-1], from_attributes=True)

        train_step = await self.model.get_step() # type: TrainableModel
        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = answer_reward(sample, ans_message)# if train_step > 7 else 0.0
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)# * 0.1
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
                "is_correct": int(verify(answer, agent_answer)),
                "gave_answer": int(num_answers > 0),
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "problem": problem,
                "answer": answer,
                "agent_answer": agent_answer,
                "item_difficulty": sample["item_difficulty"],
                "index": sample["index"],
            }
        )

        return trajectory



class Easy2HardTrainer_Multi(Trainer):
    include_histories: bool = False
    def __init__(
        self,
        model: art.Model,
        vllm_router: VllmRouter,
        path_config: PathConfig,
        prompt_config: PromptConfig,
        stop_criteria: StopCriteria,
        default_kwargs: dict | None = None,
        buffer_ratio: float = 0.33,
    ):
        super().__init__(model, vllm_router, path_config, prompt_config, stop_criteria, default_kwargs)
        train_logs_path = output_dirs.get_trajectories_split_dir(self.path_config.model_output_dir, "train")
        os.makedirs(train_logs_path, exist_ok=True)
        self.replay_buffer: GeneralReplayBuffer = RewardBasedDoubleQuantileReplayBuffer(
            directory=train_logs_path,
            grouping_keys=["index", "root"],
            upper_quantile=0.9,
            # buffer_size=70,  # Only keep the last N steps files
        )
        self.buffer_ratio:float = buffer_ratio

    def create_agent(self) -> AllTrajSingleTaskAgentNode:
        client = self.vllm_router.next()
        return AllTrajSingleTaskAgentNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            include_histories=self.include_histories,
        )
    
    async def rollout_step_multi(self, sample: dict, **kwargs) -> art.TrajectoryGroup:
        # Perform a forward step to get the trajectory
        # trajectory = await self.forward_step(sample, **kwargs)
        content = format_prompt(sample)
        agent = self.create_agent()
        message = ChatMessage(role="user", content=content)
        trajectory = await agent.chat(message, **kwargs)

        ans_message = ChatMessage.model_validate(trajectory.messages()[-1], from_attributes=True)

        train_step = await self.model.get_step() # type: TrainableModel
        # Compute rewards
        trajectory.reward = 0.0
        ans_reward = answer_reward(sample, ans_message)# if train_step > 7 else 0.0
        trajectory.reward += ans_reward
        fmt_reward = format_reward(trajectory)
        trajectory.reward += fmt_reward
        bhv_reward = behavior_reward(trajectory)
        trajectory.reward += bhv_reward

        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                "behavior_reward": bhv_reward,
                "is_correct": int(verify(answer, agent_answer)),
                "gave_answer": int(num_answers > 0),
            }
        )

        # Update metadata
        trajectory.metadata.update(
            {
                "problem": problem,
                "answer": answer,
                "agent_answer": agent_answer,
                "item_difficulty": sample["item_difficulty"],
                "index": sample["index"],
                "root": True,
                "replay": False,
            }
        )

        ## Update Sub Trajectories
        sub_trajectories = agent.sub_trajectories

        for sub_traj in sub_trajectories:
            sub_traj.reward = format_reward(sub_traj) + ans_reward * 0.05
            sub_traj.metadata.update({"root": False, "index": sample["index"], "replay": False})

        return art.TrajectoryGroup([trajectory] + sub_trajectories)

    async def rollout(
        self,
        dataset: list[dict],
        group_size: int,
        max_exceptions: int | float = 0,
        pbar_desc: str = "rollout",
        **kwargs,
    ) -> list[art.TrajectoryGroup]:
        """
        Performs a grouped rollout for the given dataset, generating a trajectory for each sample.
        The dataset is split into groups of size `group_size` defined in the training configuration.

        Args:
            dataset (list[dict]): List of samples to process.
            group_size (int): Number of rollouts for each sample.
            max_exceptions (int | float): Maximum number of failed trajectories to allow before failing.
                If float, then it is treated as a fraction of the dataset size.
            pbar_desc (str): Description for the progress bar.
            **kwargs: Additional keyword arguments for the rollout step.

        Returns:
            list[art.TrajectoryGroup]: List of trajectory groups generated by the model for the dataset.
        """
        if not kwargs:
            kwargs = self.default_kwargs

        groups = []
        for i, sample in enumerate(dataset):
            group = [(self.rollout_step_multi(sample, **kwargs)) for _ in range(group_size)]
            groups.extend(group)

        trajectories = []
        # for i, group in enumerate(groups):
        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc=pbar_desc,
            max_exceptions=max_exceptions,
            pbar_total_completion_tokens=False,
        )

        has_sub_trajectories:bool = False
        # Rearrange the trajectories into groups by sample_id and root status
        sample_groups:dict[tuple[int, bool], list[art.Trajectory]] = {}
        for trajectory_group in trajectory_groups:
            for trajectory in trajectory_group.trajectories:
                sample_id:int = trajectory.metadata.get("index", None)
                root:bool = trajectory.metadata.get("root", True)
                key = (sample_id, root)
                if key not in sample_groups:
                    sample_groups[key] = []
                sample_groups[key].append(trajectory)
                if root is False:
                    has_sub_trajectories = True
        
        trajectory_groups_list:list[art.TrajectoryGroup] = []
        for (sample_id, root), trajectories in sample_groups.items():
            trajectory_groups_list.append(art.TrajectoryGroup(trajectories))

        if "train" in pbar_desc: #TODO: better way to identify training
            new_files = self.replay_buffer.update_trajectories()

            for traj_group in trajectory_groups_list:
                # group_key = tuple(traj_group.trajectories[0].metadata[k] for k in self.replay_buffer.grouping_keys)
                group_key = self.replay_buffer._get_grouping_key(traj_group.trajectories[0])
                n = max(1, int(len(traj_group.trajectories) * self.buffer_ratio))
                replay_buffer = self.replay_buffer.sample_group(group_key=group_key, n=n)
                for traj in replay_buffer: # mark as replay
                    traj.metadata["replay"] = True
                traj_group.trajectories.extend(replay_buffer) # add replay to the group

        return trajectory_groups_list
