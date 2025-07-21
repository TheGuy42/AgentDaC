import torch
import numpy as np
import os
from dotenv import load_dotenv
import random
import re
from datasets import Dataset
from openai import AsyncOpenAI
from datetime import datetime

import art
from openpipe.client import OpenPipe
from art.local import LocalBackend
from art.dev.model import (
    InitArgs,
    EngineArgs,
    PeftArgs,
    TrainerArgs,
    InternalModelConfig,
)
from art.utils.output_dirs import (
    get_output_dir_from_model_properties,
    get_step_checkpoint_dir,
)
from vllm_client import VllmClient, ArtVLLMClient, VllmRouter
from wandb.sdk.wandb_run import Run


class Trainer:
    def __init__(
        self,
        model_name: str,
        model_config: InternalModelConfig,
        project_name: str,
        run_name: str | None = None,
        backend: LocalBackend = LocalBackend(path="./.art"),
        WANDB_API_KEY: str = "",
        OPENPIPE_API_KEY: str = "",
        seed: int = 42,
        gpu: list[int] = [0],
        vllm_server_ports: list[int] = [],
    ):
        load_dotenv()
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        os.environ["OPENPIPE_API_KEY"] = OPENPIPE_API_KEY
        os.environ["OMP_NUM_THREADS"] = "1"  # Set OMP_NUM_THREADS to 1 to avoid multithreading issues with vLLM
        os.environ["NCCL_CUMEM_ENABLE"] = "1"  # Enable NCCL cumulative memory management

        self.model_name = model_name
        self.model_config = model_config
        self.project_name = project_name
        ## Backend configuration
        self.op_client = OpenPipe()
        print("OpenPipe client initialized")
        random.seed(seed)
        self.backend = backend

        self.run_name = run_name if run_name is not None else self._generate_run_name()
        self.output_dir = get_output_dir_from_model_properties(
            self.project_name, name=self.run_name, art_path=backend._path
        )
        self.seed = seed
        self.gpu = gpu

        ## gpu configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, self.gpu))

        self.model: art.TrainableModel = None
        ## Initialize the vllm router with the provided VLLM server ports
        self.vllm_router: VllmRouter = VllmRouter(
            vllm_clients=[
                VllmClient(
                    port=port,
                    base_model=model_name,
                )
                for port in vllm_server_ports
            ]
        )

    async def load_model(self, art_port: int = 8000) -> None:
        print(f"Loading model {self.model_name} with config {self.model_config}")
        self.model = art.TrainableModel(
            name=self.run_name,
            project=self.project_name,
            base_model=self.model_name,
            _internal_config=self.model_config,
        )
        # counter = 0
        # sucess = False
        # while not sucess and counter < 5:
        #     try:
        #         await self.model.register(self.backend)
        #         sucess = True
        #     except TimeoutError as e:
        #         print(f"TimeoutError:\n Retrying({counter}) to register the model {self.model_name}...")
        #         counter += 1
        await self.model.register(self.backend)
        # self.model.inference_base_url = self.model.inference_base_url.replace(":8000", f":{art_port}")

    def _generate_run_name(self) -> str:
        """
        Generate a run name based on the model name and current date.
        """
        model_name = self.model_name.split("/")[-1]
        # Get the current date in MM_DD_HH_MM format
        date_str = datetime.now().strftime("%m_%d_%H_%M")
        # Combine model name and date to create a unique run name
        run_name = f"{model_name}_{date_str}"
        output_dir = get_output_dir_from_model_properties(self.project_name, name=run_name, art_path=self.backend._path)
        if os.path.exists(output_dir):
            raise ValueError(f"Output directory {output_dir} already exists. try again in a minute.")
        return run_name

    def get_wandb_run(self) -> Run | None:
        """
        Get the wandb run associated with this trainer. If not existent then start one.
        This will return None if the model has not been loded yet or if wandb key not provided.
        Returns:
            Run: The wandb run object if it exists, otherwise None.
        """
        if self.model is None:
            print("Model is not loaded yet. Cannot get wandb run.")
            return None
        return self.backend._get_wandb_run(self.model)

    def update_wandb_config(self, config: dict) -> None:
        """
        Update the wandb configuration for the current run.
        Args:
            config (dict): The configuration dictionary to update.
        """
        wandb_run = self.get_wandb_run()
        if wandb_run is not None:
            wandb_config = wandb_run.config.as_dict()
            wandb_config.update(config)
            for key, value in wandb_config.items():
                wandb_run._set_config_wandb(key, value)
        else:
            print("Wandb run is not available. Cannot update config.")

    async def train(
        self,
        epochs: int = 1,
        n_rollouts: int = 1,
        n_groups: int = 1,
    ):
        """
        Train the model for a specified number of epochs. This method handles the vllm client management,
        loading and unloading of lora adapters, and the training process itself.
        Training logic should be implemented in the `_train` method.
        Args:
            epochs (int): The number of epochs to train the model.
            n_rollouts (int): The number of rollouts to perform per epoch.
            n_groups (int): The number of groups to use for training.
        """
        self.vllm_router.unload_all_loras()  # Unload all lora adapters before starting the training
        self.vllm_router.add_client(ArtVLLMClient(self.model))

        prev_step_checkpoint_dir = None
        step_checkpoint_dir = None

        for i in range(await self.model.get_step(), epochs):
            print(f"Starting step {i} for model {self.model.name}")
            # Load and unload lora adapters if needed
            prev_step_checkpoint_dir = step_checkpoint_dir
            step_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, i) if i > 0 else None
            step_checkpoint_dir = (
                step_checkpoint_dir.replace("./art/", ".art/") if step_checkpoint_dir is not None else None
            )

            if prev_step_checkpoint_dir is not None:
                print(f"Unloading lora")
                if not self.vllm_router.unload_lora(self.run_name):
                    print(f"Failed to unload lora from {prev_step_checkpoint_dir}, using base model {self.model_name}")
                    break
            if step_checkpoint_dir is not None:
                print(f"Loading lora from {step_checkpoint_dir}")
                if not self.vllm_router.load_lora(self.run_name, step_checkpoint_dir):
                    print(f"Failed to load lora from {step_checkpoint_dir}, using base model {self.model_name}")
                    break

            await self._train(
                epoch=i,
                n_rollouts=n_rollouts,
                n_groups=n_groups,
                vllm_router=self.vllm_router,  # can be accessed from the class instance but passing for explicity
            )

    async def _train(
        self,
        epoch: int,
        n_rollouts: int,
        n_groups: int,
        vllm_router: VllmRouter,
    ):
        """
        Placeholder for the training logic.
        This method should be implemented to handle the training process,
        including data preparation, model training, and evaluation.
        Args:
            epoch (int): The current training epoch.
            n_rollouts (int): The number of rollouts to perform.
            n_groups (int): The number of groups for training.
            vllm_router (VllmRouter): The router to manage VLLM clients
        """
        raise NotImplementedError("_train is not implemented yet.")
