from __future__ import annotations
import openai
import httpx
import asyncio
import os
import pathlib

import art
from art.openai import patch_openai
from src.utils.logging import create_logger


logger = create_logger(__name__)


class VllmClient:
    def __init__(
        self,
        base_url: str | httpx.URL,
        base_model: str,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = httpx.Timeout(timeout=1200, connect=5.0),
    ):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                api_key = "default"
                logger.info(f"No OpenAI API key provided, using '{api_key}' key.")

        if model_name is None:
            model_name = base_model

        self.base_url = base_url
        self.base_model = base_model
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

        self.http_client = openai.DefaultAsyncHttpxClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            limits=httpx.Limits(max_connections=100_000, max_keepalive_connections=100_000),
        )

        self.openai_client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=self.http_client,
            max_retries=3,
        )

        self.openai_client = patch_openai(self.openai_client)

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the HTTP client and cleanup resources"""
        try:
            if hasattr(self.http_client, 'aclose'):
                await self.http_client.aclose()
            logger.info(f"[{self.base_url}] HTTP client closed")
        except Exception as e:
            logger.warning(f"[{self.base_url}] Error closing HTTP client: {e}")

    @staticmethod
    def from_connection(
        port: int,
        base_model: str,
        model_name: str | None = None,
        host: str = "0.0.0.0",
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = httpx.Timeout(timeout=1200, connect=5.0),
        **kwargs,
    ) -> VllmClient:
        return VllmClient(
            base_url=f"http://{host}:{port}/v1",
            base_model=base_model,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

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
    @staticmethod
    def from_art_model(art_model: art.TrainableModel | art.Model, **kwargs) -> VllmClient:
        """
        Create a VllmClient instance from an ART Model.
        """
        client = art_model.openai_client()

        if isinstance(art_model, art.TrainableModel):
            return ArtClient(
                base_url=client.base_url,
                base_model=art_model.base_model,
                model_name=art_model.get_inference_name(),
                api_key=client.api_key,
                timeout=client.timeout,
                **kwargs,
            )
        else:
            return ArtClient(
                base_url=client.base_url,
                base_model=art_model.get_inference_name(),
                model_name=None,
                api_key=client.api_key,
                timeout=client.timeout,
                **kwargs,
            )

    async def load_lora(self, lora_name: str, lora_path: str):
        return  # No-op for ArtClient, as it uses ART's LORA management

    async def unload_lora(self, lora_name: str):
        return  # No-op for ArtClient, as it uses ART's LORA management


class VllmRouter:
    """
    An iterator class that rotates through a list of VllmClient instances.
    """

    def __init__(self, clients: list[VllmClient] | None = None):
        if clients is None:
            clients = []

        self.clients: list[VllmClient] = clients
        self.idx = 0

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

    async def close_all(self) -> None:
        """
        Close all clients and cleanup resources.
        """
        tasks = [client.close() for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)
