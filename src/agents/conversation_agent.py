from __future__ import annotations
import asyncio
from typing import Type

from openai.types.chat import ChatCompletion
from art.trajectories import Trajectory, History

from src.agents.base import BaseAgent
from src.agents.strategies.base import ParseStrategy, AgentTurn, TurnAction
from src.agents.strategies.marker_strategy import MarkerParseStrategy
from src.openai_types import Message, UserMessage
from src.utils.visualize import trajectory_string, message_string
from src.utils.logging import create_logger
from src.configs.prompts import get_prompt

logger = create_logger(__name__)


class ConversationAgent(BaseAgent):
    """
    Unified conversation agent that uses pluggable parsing strategies.
    This eliminates code duplication between JSON, Regex, and Marker agents.
    """
    
    def __init__(
        self,
        openai_client,
        model_name: str,
        prompt_config,
        decomp_config,
        parse_strategy: ParseStrategy,
        current_depth: int = 0,
        additional_histories: bool = False,
        agent_class: Type[BaseAgent] | None = None,
    ):
        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            prompt_config=prompt_config,
            decomp_config=decomp_config,
            current_depth=current_depth,
            additional_histories=additional_histories,
        )
        self.parse_strategy = parse_strategy
        self.agent_class = agent_class or ConversationAgent
        
    def _create_subagent(self) -> BaseAgent:
        """Create a subagent of the same type with incremented depth."""
        return self.agent_class(
            openai_client=self.openai_client,
            model_name=self.model,
            prompt_config=self.prompt_config,
            decomp_config=self.decomp_config,
            parse_strategy=self.parse_strategy,
            current_depth=self.current_depth + 1,
            additional_histories=False,  # No support for recursive histories yet
            agent_class=self.agent_class,
        )
    
    async def call(self, messages: list[Message], **kwargs) -> ChatCompletion:
        """Call OpenAI API with strategy-specific modifications."""
        # Let the parse strategy modify the call kwargs
        allowed_actions = self.parse_strategy.get_allowed_actions(
            self.decomp_config, self.current_depth
        )
        kwargs = self.parse_strategy.prepare_call_kwargs(allowed_actions, **kwargs)
        
        return await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            **kwargs,
        )

    async def chat(
        self,
        prompt: Message,
        verbose: bool = False,
        **kwargs,
    ) -> Trajectory:
        """Unified chat method that works with any parsing strategy."""
        if prompt.get("role") != "user":
            logger.warning(f"Prompt role is expected to be 'user', but got {prompt.get('role')}.")

        self.trajectory.messages_and_choices.append(prompt)

        # Store the initial prompt in metadata for reference
        content = prompt.get("content")
        if isinstance(content, str):
            self.metadata["prompt"] = content

        if verbose:
            print(trajectory_string(self.trajectory, indent=self.current_depth))

        # Initialize metrics
        self.metrics.setdefault("direct_thinks", 0)
        self.metrics.setdefault("total_thinks", 0)

        # Handle marker strategy differently due to its unique multi-task approach
        if isinstance(self.parse_strategy, MarkerParseStrategy):
            return await self._marker_chat_loop(verbose, **kwargs)
        else:
            return await self._standard_chat_loop(verbose, **kwargs)

    async def _standard_chat_loop(self, verbose: bool, **kwargs) -> Trajectory:
        """Standard chat loop for JSON and Regex strategies."""
        while True:
            # Get allowed actions and call the model
            allowed_actions = self.parse_strategy.get_allowed_actions(
                self.decomp_config, self.current_depth
            )
            
            completion = await self.call(self.trajectory.messages(), **kwargs)
            choice = completion.choices[0]
            self.trajectory.messages_and_choices.append(choice)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage is not None:
                self.metrics["total_tokens"] = completion.usage.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Parse the response
            try:
                assistant_msg = self.trajectory.messages()[-1]
                content = assistant_msg.get("content", "")
                turn = self.parse_strategy.parse_response(content, allowed_actions)
            except Exception as e:
                logger.warning(f"Failed to parse model output: {e}")
                turn = AgentTurn(action=TurnAction.ERROR, text="", raw=content)
                break

            # Handle the action
            if turn.action == TurnAction.ANSWER:
                break
            elif turn.action == TurnAction.THINK:
                self.metrics["direct_thinks"] += 1
                self.metrics["total_thinks"] += 1
                self.decomp_config.update_round(num_tasks=0)
            elif turn.action == TurnAction.ISSUE_TASK:
                await self._handle_task_delegation(turn, verbose, **kwargs)
            elif turn.action == TurnAction.ERROR:
                break
            else:
                raise ValueError(f"Unhandled action: {turn.action}")

        # Update final stats
        self.metrics["response_completed"] = (
            choice.finish_reason != "length" and turn.action == TurnAction.ANSWER
        )
        self.trajectory.finish()
        return self.trajectory

    async def _marker_chat_loop(self, verbose: bool, **kwargs) -> Trajectory:
        """Special chat loop for marker strategy to handle multi-task delegation."""
        should_break = False

        while True:
            # Call the OpenAI API to get a response
            completion = await self.call(self.trajectory.messages(), **kwargs)
            choice = completion.choices[0]
            self.trajectory.messages_and_choices.append(choice)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["direct_calls"] += 1
            if completion.usage is not None:
                self.metrics["total_tokens"] = completion.usage.total_tokens

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

            # Extract tasks from the response using marker strategy
            tasks_inputs = self.parse_strategy.parse_tasks(self.trajectory.messages()[-1])

            # If no tasks to delegate then last message
            if should_break or len(tasks_inputs) == 0:
                break

            if self._should_stop_marker():
                mock_answer = get_prompt(self.prompt_config.tasks_depleted)
                if mock_answer is None:
                    break
                should_break = True
                tasks_answers = [mock_answer] * len(tasks_inputs)
            else:
                # Process tasks concurrently
                tasks_answers = await asyncio.gather(
                    *[self._process_marker_task(task, verbose, **kwargs) for task in tasks_inputs]
                )

            # Update metrics
            self.metrics["direct_tasks"] += len(tasks_inputs)
            self.metrics["total_tasks"] += len(tasks_inputs)
            self.decomp_config.update_round(num_tasks=len(tasks_inputs))

            # Create unified response
            from src.agents.marker_agent.markers import Markers
            tasks_answers = [f"{Markers.ANS_START} {ans} {Markers.ANS_END}" for ans in tasks_answers]
            unified_answer = "\n".join(tasks_answers)
            joined_message = UserMessage(role="user", name="sub-agent", content=unified_answer)
            self.trajectory.messages_and_choices.append(joined_message)

            if verbose:
                print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

        # Update final stats
        self.metrics["response_completed"] = choice.finish_reason != "length"
        self.trajectory.finish()
        return self.trajectory

    async def _handle_task_delegation(self, turn: AgentTurn, verbose: bool, **kwargs):
        """Handle task delegation for standard strategies."""
        sub_agent = self._create_subagent()
        task = UserMessage(role="user", content=turn.text)
        task_answer = await sub_agent.answer(task, verbose, **kwargs)
        task_response = UserMessage(role="user", name="sub-agent", content=task_answer)
        self.trajectory.messages_and_choices.append(task_response)

        if self.additional_histories:
            history = History(messages_and_choices=sub_agent.trajectory.messages_and_choices)
            self.trajectory.additional_histories.append(history)

        if verbose:
            print(message_string(self.trajectory.messages()[-1], indent=self.current_depth))

        # Update metrics from sub-agent
        self.metrics["direct_tasks"] += 1
        self.metrics["total_tasks"] += 1 + sub_agent.metrics["total_tasks"]
        self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
        self.metrics["total_thinks"] += sub_agent.metrics["total_thinks"]
        self.metrics["max_depth"] = max(
            1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"]
        )
        self.decomp_config.update_round(num_tasks=1)

    async def _process_marker_task(self, task: UserMessage, verbose: bool, **kwargs) -> str:
        """Process a single task for marker strategy."""
        sub_agent = self._create_subagent()
        answer = await sub_agent.answer(task, verbose, **kwargs)

        if self.additional_histories:
            history = History(messages_and_choices=sub_agent.trajectory.messages_and_choices)
            self.trajectory.additional_histories.append(history)

        # Update metrics from sub-agent
        self.metrics["total_tasks"] += sub_agent.metrics["total_tasks"]
        self.metrics["total_calls"] += sub_agent.metrics["total_calls"]
        self.metrics["max_depth"] = max(
            1 + sub_agent.metrics["max_depth"], self.metrics["max_depth"]
        )
        return answer

    def _should_stop_marker(self) -> bool:
        """Check if marker strategy should stop delegating tasks."""
        dc = self.decomp_config
        if self.current_depth >= dc.max_depth:
            return True
        if dc.total_tasks >= dc.max_tasks:
            return True
        if dc.total_rounds >= dc.max_rounds:
            return True
        return False

    @staticmethod
    def parse_answer(message: Message) -> str:
        """Parse answer using appropriate strategy - this will be overridden by subclasses."""
        if message["role"] != "assistant":
            logger.error(f"Expected message role 'assistant', got {message['role']}")
            raise ValueError("Message role must be 'assistant' to extract answer.")

        content = message.get("content")
        if not isinstance(content, str):
            logger.error(f"Expected message content to be a string, got {type(content)}")
            raise ValueError("Message content must be a string.")

        return content  # Default implementation