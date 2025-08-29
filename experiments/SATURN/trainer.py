from src.dac_agent import AgentNode
from src.dac_agent_single import SingleAgentNode
from src.dac_agent_all_traj import AllTrajSingleTaskAgentNode
from src.trainer import Trainer
from src.dac_agent import ChatMessage
from src.configs.markers import Markers
from src.utils.text import extract_answer, extract_between

from experiments.general_rewards import format_reward, behavior_reward
from experiments.SATURN.rewards import answer_reward, verify
from experiments.SATURN.format import format_prompt

import art


class SATURN_Trainer(Trainer):
    def create_agent(self) -> SingleAgentNode:
        client = self.vllm_router.next()
        return SingleAgentNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            include_histories=True,
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
        fmt_reward = format_reward(trajectory) * 0.1
        trajectory.reward += fmt_reward
        # bhv_reward = behavior_reward(trajectory)
        # trajectory.reward += bhv_reward

        problem = format_prompt(sample)
        answer = sample["answer"].strip()
        agent_answer = extract_answer(ans_message.content)
        num_answers = len(extract_between(ans_message.content, Markers.ANSWER_START, Markers.ANSWER_END))

        # Update metrics
        trajectory.metrics.update(
            {
                "answer_reward": ans_reward,
                "format_reward": fmt_reward,
                # "behavior_reward": bhv_reward,
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
                # "root": True,
            }
        )


        ## Update Sub Trajectories
        # if hasattr(agent, "sub_trajectories") and agent.sub_trajectories:
        #     for sub_traj in agent.sub_trajectories:
        #         sub_traj.reward = format_reward(sub_traj) + ans_reward
        #         sub_traj.metadata.update({"root": False})

        return trajectory


class SATURN_Trainer_Multi(Trainer):
    def create_agent(self) -> AllTrajSingleTaskAgentNode:
        client = self.vllm_router.next()
        return AllTrajSingleTaskAgentNode(
            model_name=self.model.get_inference_name(),
            openai_client=client.openai_client,
            prompt_config=self.prompt_config,
            stop_criteria=self.stop_criteria,
            include_histories=False,
        )
    
    async def rollout_step_multi(self, sample: dict, sample_id:int, **kwargs) -> art.TrajectoryGroup:
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
                "sample_id": sample_id,
                "root": True,
            }
        )

        ## Update Sub Trajectories
        sub_trajectories = agent.sub_trajectories
        # if len(sub_trajectories) == 0 and trajectory.metrics.get("total_tasks", 0) > 0:
        #     print("#############################################################")
        #     print("No sub-trajectories found in the rollout.")
        #     print("#############################################################")

        for sub_traj in sub_trajectories:
            sub_traj.reward = format_reward(sub_traj) + ans_reward * 0.3
            sub_traj.metadata.update({"root": False, "sample_id": sample_id})

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
            group = [(self.rollout_step_multi(sample, i, **kwargs)) for _ in range(group_size)]
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
                sample_id:int = trajectory.metadata.get("sample_id", None)
                root:bool = trajectory.metadata.get("root", True)
                key = (sample_id, root)
                if key not in sample_groups:
                    sample_groups[key] = []
                sample_groups[key].append(trajectory)
                if root is False:
                    has_sub_trajectories = True
        
        # if not has_sub_trajectories:
        #     print("#############################################################")
        #     print("No sub-trajectories found in the rollout.")
        #     print("#############################################################")

        trajectory_groups_list:list[art.TrajectoryGroup] = []
        for (sample_id, root), trajectories in sample_groups.items():
            trajectory_groups_list.append(art.TrajectoryGroup(trajectories))

        return trajectory_groups_list
