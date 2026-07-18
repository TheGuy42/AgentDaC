from __future__ import annotations

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

from src.trajectory import Trajectory
from src.utils.logging import create_logger
from src.inference import OAIResponse, InferenceResponse

logger = create_logger(__name__)


def _extract_attributes(response: OAIResponse) -> tuple[list[int], list[int]]:

    resp = response.openai_response
    response_token_ids: list[int]
    prompt_token_ids: list[int]

    if hasattr(resp.choices[0], "token_ids"):
        # This is the original object structure from OpenAI
        response_token_ids = getattr(resp.choices[0], "token_ids")

    elif hasattr(resp.choices[0], "provider_specific_fields") and ("token_ids" in resp.choices[0].provider_specific_fields):  # type: ignore[attr-defined]
        # When manually upgrading to vllm==0.11.0 and verl==0.6.1 (latest stable versions)
        # then AGL patch stops working and the token_ids are placed here
        response_token_ids = resp.choices[0].provider_specific_fields["token_ids"]  # type: ignore[attr-defined]

    elif hasattr(resp, "response_token_ids"):
        # This attr appears in the patched response object by AGL
        # See `agentlightning.instrumentation.vllm.ChatCompletionResponsePatched`
        response_token_ids = getattr(resp, "response_token_ids")[0]

    else:
        logger.error(f"Response object missing expected token id attributes: {resp.model_dump()}")
        raise ValueError("Unable to extract response token ids from response object")

    if hasattr(resp, "prompt_token_ids"):
        # This attr appears in the original and the patched response object by AGL
        # See `agentlightning.instrumentation.vllm.ChatCompletionResponsePatched`
        prompt_token_ids = getattr(resp, "prompt_token_ids")

    else:
        logger.error(f"Response object missing expected token id attributes: {resp.model_dump()}")
        raise ValueError("Unable to extract prompt token ids from response object")

    return prompt_token_ids, response_token_ids


def convert_trajectory(
    trajectory: Trajectory,
    prompt_length: int,
    response_length: int,
) -> AgentLoopOutput:
    """
    Convert a finished `Trajectory` to a VERL `AgentLoopOutput`.

    Args:
        trajectory (Trajectory): The finished trajectory to convert.
        prompt_length (int): The maximum length of the prompt in tokens.
        response_length (int): The maximum length of the response in tokens.
            If the response is longer than this, it will be truncated.
    """
    if len(trajectory.histories) > 0:
        logger.warning("Trajectory has additional histories; They are not yet supported and will be ignored in the span conversion.")
        logger.info("To convert and train on additional histories, please pass them explicitly.")

    responses = [r for r in trajectory.messages_and_responses if isinstance(r, OAIResponse)]
    if len(responses) == 0:
        raise ValueError("Trajectory has no OAIResponses; cannot convert to AgentLoopOutput.")

    # STRUCTURE:
    # prompt_ids - all the tokens before the first assistant token
    # response_ids - all the tokens after the first assistant token
    # response_mask - only over response_ids, 1 for model-generated tokens and 0 otherwise

    prompt_ids, _ = _extract_attributes(responses[0])
    if len(prompt_ids) > prompt_length:
        raise ValueError(
            f"Initial prompt has {len(prompt_ids)} tokens > rollout.prompt_length={prompt_length}. "
            f"Refusing to left-truncate the prompt (the model generated under the full prompt, so "
            f"Increase prompt_length or shorten the prompt."
        )

    ctx_ids, rsp_ids = _extract_attributes(responses[-1])
    traj_ids = ctx_ids + rsp_ids
    traj_mask: list[int] = []

    for i, resp in enumerate(responses):
        p_ids, r_ids = _extract_attributes(resp)
        n_zeros, n_ones = len(p_ids) - len(traj_mask), len(r_ids)
        traj_mask += [0] * n_zeros + [1] * n_ones

        # Now ensure that [p_ids + r_ids] are a prefix of traj_ids
        if p_ids + r_ids != traj_ids[: len(p_ids) + len(r_ids)]:
            raise ValueError(
                f"Turn {i}: the tokenized prompt is not a continuation of the trajectory tokens "
                f"(chat-template re-tokenization drift). Refusing to build a corrupted token sequence."
            )

    response_ids = traj_ids[len(prompt_ids) :]
    response_mask = traj_mask[len(prompt_ids) :]

    # Truncate the response ids and mask to the configured response_length
    if len(response_ids) > response_length:
        logger.warning(
            f"Response has {len(response_ids)} tokens > rollout.response_length={response_length}. "
            f"Truncating the response to the first {response_length} tokens. "
            f"Consider increasing response_length or shortening the model's output."
        )
        response_ids = response_ids[:response_length]
        response_mask = response_mask[:response_length]

    metrics = {k: float(v) for k, v in trajectory.metrics.items()}
    metrics["is_degenerate"] = 0.0

    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        reward_score=float(trajectory.reward),
        num_turns=len(responses),
        metrics=AgentLoopMetrics(),
        extra_fields={"custom_metrics": metrics},
    )


def degenerate_output(trajectory: Trajectory, tokenizer) -> AgentLoopOutput:
    prompt_ids = tokenizer.encode("dummy input")
    response_ids = tokenizer.encode("dummy output")

    responses = [r for r in trajectory.messages_and_responses if isinstance(r, InferenceResponse)]
    metrics = {"is_degenerate": 1.0}

    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=[0] * len(response_ids),
        reward_score=0.0,
        num_turns=len(responses),
        metrics=AgentLoopMetrics(),
        extra_fields={"custom_metrics": metrics},
    )
