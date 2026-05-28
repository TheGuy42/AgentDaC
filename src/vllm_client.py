from __future__ import annotations
import openai
import httpx
import asyncio
import pathlib

import art
from src.utils.logging import create_logger


logger = create_logger(__name__)


class VllmClient:
    def __init__(
        self,
        openai_client: openai.AsyncOpenAI,
        base_model: str,
        model_name: str | None = None,
    ):
        """
        Args:
            openai_client (openai.AsyncOpenAI): The OpenAI client to use for API
            base_model (str): The openai model name of the main weights
            model_name (str, optional): The LORA adapter name to use for inference, overriding the base model.
        """
        if model_name is None:
            model_name = base_model

        self.openai_client = openai_client
        self.http_client = openai_client._client

        self.base_url = openai_client.base_url
        self.base_model = base_model
        self.model_name = model_name

    @staticmethod
    def from_connection(
        port: int,
        base_model: str,
        model_name: str | None = None,
        host: str = "0.0.0.0",
        api_key: str | None = None,
        **kwargs,
    ) -> VllmClient:
        kwargs.setdefault("max_retries", 3)
        kwargs.setdefault("timeout", httpx.Timeout(timeout=1200, connect=5.0))

        openai_client = openai.AsyncOpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key, **kwargs)

        return VllmClient(
            openai_client=openai_client,
            base_model=base_model,
            model_name=model_name,
            **kwargs,
        )

    async def close(self):
        try:
            await self.openai_client.close()
            await self.http_client.aclose()
        except Exception as e:
            logger.error(f"[{self.base_url}] Error closing HTTP client: {e}")

    def get_inference_name(self) -> str:
        return self.model_name

    async def verify_connection(self) -> bool:
        """
        Verify if the model is available on the VLLM server.
        """
        try:
            await self.openai_client.models.retrieve(self.get_inference_name())
        except openai.APIConnectionError:
            return False
        except Exception as e:
            logger.error(f"[{self.base_url}] Failed to connect to VLLM server: {e}")
            return False
        return True

    @art.utils.retry(max_attempts=3)
    async def load_lora(self, lora_name: str, lora_path: str):
        payload = {"lora_name": lora_name, "lora_path": lora_path}
        resp = await self.http_client.post("load_lora_adapter", json=payload)
        resp.raise_for_status()
        self.model_name = lora_name  # Update the inference name to the loaded LORA

        lora_path = pathlib.Path(lora_path).relative_to(pathlib.Path.cwd()).as_posix()
        logger.info(f"[{self.base_url}] Loaded LORA adapter: {lora_name} at path: {lora_path}")

    @art.utils.retry(max_attempts=3)
    async def unload_lora(self, lora_name: str):
        if lora_name == self.base_model:
            logger.warning(f"[{self.base_url}] Cannot unload the base model. Use a different LORA name.")
            return

        payload = {"lora_name": lora_name}
        response = await self.http_client.post("unload_lora_adapter", json=payload)
        response.raise_for_status()
        logger.info(f"[{self.base_url}] Unloaded LORA adapter: {lora_name}")

        # Reset the inference name to the base model
        if lora_name == self.get_inference_name():
            self.model_name = self.base_model

    async def get_model_list(self) -> list[openai.types.Model]:
        """
        Get the list of models available on the VLLM server.
        """
        result = await self.openai_client.models.list()
        model_list = result.data

        if len(model_list) == 0:
            logger.warning(f"[{self.base_url}] No models found on the VLLM server.")

        return model_list

    async def unload_all_loras(self):
        unload_tasks = []
        for model in await self.get_model_list():
            name = model.id
            parent = getattr(model, "parent", None)
            if name != self.base_model and parent == self.base_model:
                unload_tasks.append(self.unload_lora(model.id))
        await asyncio.gather(*unload_tasks)


class ArtClient(VllmClient):
    def __init__(
        self,
        art_model: art.TrainableModel,
        **kwargs,
    ):
        self.art_model = art_model
        super().__init__(
            openai_client=art_model.openai_client(),
            base_model=art_model.base_model,
            model_name=art_model.get_inference_name(),
            **kwargs,
        )

    async def load_lora(self, lora_name: str, lora_path: str):
        # Update the inference name to reflect the current ART model state
        self.model_name = self.art_model.get_inference_name()
        return

    async def unload_lora(self, lora_name: str):
        # Update the inference name to reflect the current ART model state
        self.model_name = self.art_model.get_inference_name()
        return

    async def close(self):
        return  # No-op for ArtClient, as ART manages the client lifecycle


class VllmRouter:
    """
    An iterator class that rotates through a list of VllmClient instances.
    """

    def __init__(self, clients: list[VllmClient] | None = None):
        if clients is None:
            clients = []

        self.clients: list[VllmClient] = clients
        self.idx = 0

    async def close(self) -> None:
        """
        Close all HTTP clients in the router.
        """
        tasks = [client.close() for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)

    def __len__(self) -> int:
        return len(self.clients)

    def next(self) -> VllmClient:
        if len(self) == 0:
            raise ValueError("No clients available in the router.")

        client = self.clients[self.idx]
        self.idx = (self.idx + 1) % len(self)
        return client

    def append(self, client: VllmClient):
        """
        Add a new VLLMClient to the router.
        """
        self.clients.append(client)

    def current(self) -> VllmClient:
        """
        Get the current client in the round-robin rotation.
        """
        if len(self) == 0:
            raise ValueError("No clients available in the router.")

        client = self.clients[self.idx]
        return client

    async def load_lora(self, lora_name: str, lora_path: str):
        """
        Load a LORA adapter on all clients.
        """
        tasks = [client.load_lora(lora_name, lora_path) for client in self.clients]
        await asyncio.gather(*tasks)

    async def unload_lora(self, lora_name: str):
        """
        Unload a LORA adapter on all clients.
        """
        tasks = [client.unload_lora(lora_name) for client in self.clients]
        await asyncio.gather(*tasks)

    async def unload_all_loras(self) -> None:
        """
        Unload all LORA adapters from all clients.
        """
        tasks = [client.unload_all_loras() for client in self.clients]
        await asyncio.gather(*tasks)
