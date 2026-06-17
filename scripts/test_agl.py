# fix imports
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# os.chdir(module_path)
print(f"Current Working Directory: {os.getcwd()}")

import agentlightning as agl
from src.agents import DummyAgent
from openai import AsyncOpenAI
from src.configs import PromptConfig
from experiments.memorization.rewards import answer_reward
from experiments.memorization.format import format_prompt
import random
from experiments.experiment_runner import Dataset


# unset ROCR_VISIBLE_DEVICES env var
import os
if "ROCR_VISIBLE_DEVICES" in os.environ:
    del os.environ["ROCR_VISIBLE_DEVICES"]
    


VERL_CONFIG = {
    "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
    "trainer": {
        "project_name": "memorization_dummy",
        "experiment_name": "memorization_dummy",
        "resume_from_path": None,
        "nnodes": 1,
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "test_freq": 5,
        "save_freq": -1,
        "total_epochs": 10,
        "max_actor_ckpt_to_keep": 1,
        "max_critic_ckpt_to_keep": 1,
        "logger": ["console", "wandb"],
    },
    "data": {
        "train_batch_size": 4,
        "max_prompt_length": 1024,
        "max_response_length": 512,
        "truncation": "error",
        "shuffle": True,
        "train_files": None,
        "val_files": None,
        "apply_chat_template_kwargs": {"enable_thinking": False},
    },
    "actor_rollout_ref": {
        "rollout": {
            "n": 16,
            "do_sample": True,
            "temperature": 1.2,
            "val_kwargs": {"n": 1, "do_sample": False, "temperature": 1.2},
            "tensor_model_parallel_size": 1,
            "log_prob_micro_batch_size_per_gpu": 4,
            "name": "vllm",
            "dtype": "bfloat16",
            "load_format": "safetensors",
            "gpu_memory_utilization": 0.7,
            "max_model_len": None,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 1024,
            "engine_kwargs": {
                "vllm": {"generation_config": "auto", "enable_auto_tool_choice": True, "tool_call_parser": "hermes", "max_lora_rank": 64}
            },
            "multi_turn": {"enable": True, "format": "hermes"},
        },
        "actor": {
            "ppo_mini_batch_size": 2,
            "ppo_micro_batch_size_per_gpu": 2,
            "optim": {"lr": 1e-5},
            "policy_loss": {"loss_mode": "vanilla"},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0.001,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {"param_offload": False, "optimizer_offload": False, "model_dtype": "bfloat16"},
        },
        "ref": {"log_prob_micro_batch_size_per_gpu": 8, "fsdp_config": {"param_offload": False, "model_dtype": "bfloat16"}},
        "model": {
            "path": "Qwen/Qwen2.5-0.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
            "lora_rank": 16,
            "lora_alpha": 32,
            "target_modules": "all-linear",
        },
    },
    "agentlightning": {
        "trace_aggregator": {
            "level": "transition",
            "trajectory_max_prompt_length": 1024,
            "trajectory_max_response_length": 512,
            "debug": False,
            "mismatch_log_dir": "./mismatch_cases",
        }
    },
}



LABEL_KINDS = ["random", "const"]


def load_data(num_samples: int, labels: list[str], label_kind: str = "random") -> tuple[Dataset, Dataset, Dataset]:
    if label_kind not in LABEL_KINDS:
        raise ValueError(f"Invalid label_kind: {label_kind}. Must be one of {LABEL_KINDS}.")

    if label_kind == "random":
        random_labels = (labels * (num_samples // len(labels) + 1))[:num_samples]
        random.shuffle(random_labels)
    elif label_kind == "const":
        lbl = random.choice(labels)
        random_labels = [lbl] * num_samples
        print(f"Using constant label: {lbl}")
    else:
        raise ValueError(f"Invalid label_kind: {label_kind}")

    samples = [{"id": i, "answer": random_labels[i], "labels": labels} for i in range(num_samples)]
    ds_train = Dataset.from_list(samples)
    return ds_train, ds_train, ds_train




class AglAgent(agl.LitAgent):
    async def rollout_async(self, task: dict, resources: dict, rollout: agl.Rollout):
        llm = resources["main_llm"]

        if not isinstance(llm, agl.LLM):
            raise TypeError(f"Resource 'main_llm' must be an LLM, got {type(llm)}.")

        if not isinstance(rollout, agl.AttemptedRollout):
            raise TypeError(f"Expected rollout to be an AttemptedRollout, got {type(rollout)}.")

        base_url = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
        openai_client = AsyncOpenAI(base_url=base_url, api_key=llm.api_key or "EMPTY")

        dummy_agent = DummyAgent(
            openai_client=openai_client,
            model_name=VERL_CONFIG["actor_rollout_ref"]["model"]["path"],
            prompt_config=PromptConfig(mode="text"),
        )
        
        prompt = format_prompt(task)
        response = await dummy_agent.answer({"role": "user", "content": prompt})
        
        reward, parse_success = answer_reward(task, response)
        if not parse_success:
            reward -= 1.0  
        
        agl.emit_reward(reward)
        
        last_trace = self.tracer.get_last_trace()
        print(last_trace)

        
        
def main():
        
    ds_train, ds_val, ds_test = load_data(num_samples=100, labels=["A", "B", "C"], label_kind="const")

    trainer = agl.Trainer(
        n_runners=16,
        algorithm=agl.VERL(VERL_CONFIG),
    )

    agent = AglAgent()

    trainer.fit(
        agent,
        train_dataset=ds_train.to_list(),
        val_dataset=ds_val.to_list(),
    )
    
    
if __name__ == "__main__":
    main()