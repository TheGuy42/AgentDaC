import requests
from openai import AsyncOpenAI
from typing import Any, Dict, Optional
from openai.types.chat.chat_completion import ChatCompletion, Choice
from art.openai import patch_openai
from art import TrainableModel


class VllmClient:
    def __init__(
        self,
        inference_base_url: str = "http://localhost",
        port: int = 8000,
        base_model: Optional[str] = None,
        **kwargs: Any,
    ):
        self.inference_base_url = inference_base_url + f":{port}/v1"
        self.port = port
        self.base_model = base_model
        self.inference_name = base_model

        self.client = AsyncOpenAI(
            base_url=self.inference_base_url,
            api_key="default",  # Assuming no API key is needed for local server
            **kwargs,  # Additional keyword arguments for the client
        )
        self.client = patch_openai(self.client)

    def get_inference_name(self) -> str:
        return self.inference_name

    def verify_connection(self) -> bool:
        """
        Verify if the VLLM server is running and accessible.
        Returns True if the server is reachable, False otherwise.
        """
        try:
            response = requests.get(f"{self.inference_base_url}/models/{self.get_inference_name()}")
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to VLLM server: {e}")
            return False

    async def chat(
        self,
        message: Dict[str, str],
        model: str = None,
        **kwargs,
    ) -> dict:
        if model is None:
            model = self.get_inference_name()
            if not model:
                raise ValueError("Model name must be specified or set in the client.")

        response = await self.client.chat.completions.create(
            model=model,
            messages=message,
            **kwargs,  # Additional keyword arguments for the chat completion
        )
        return response

    def load_lora(self, lora_name: str, lora_path: str):
        url = f"{self.inference_base_url}/load_lora_adapter"
        headers = {"Content-Type": "application/json"}
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path,
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            self.inference_name = lora_name  # Update the inference name to the loaded LORA

            print("Request successful!")
            print("Status Code:", response.status_code)
            # print("Response JSON:", response.json())
            return {"success": True, "status_code": response.status_code}
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            if hasattr(e, "response") and e.response is not None:
                print("Response Status Code:", e.response.status_code)
                print("Response Text:", e.response.text)
            return {"success": False, "error": str(e)}

    def unload_lora(self, lora_name: str):
        url = f"{self.inference_base_url}/unload_lora_adapter"
        headers = {"Content-Type": "application/json"}
        payload = {
            "lora_name": lora_name,
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            self.inference_name = self.base_model  # Reset inference name to base model

            print("Request successful!")
            print("Status Code:", response.status_code)
            # print("Response JSON:", response.json())
            return {"success": True, "status_code": response.status_code}
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            if hasattr(e, "response") and e.response is not None:
                print("Response Status Code:", e.response.status_code)
                print("Response Text:", e.response.text)
            return {"success": False, "error": str(e)}

    def _get_vllm_model_list(self) -> list[str]:
        """
        Get the list of models available on the VLLM server.
        Returns a list of model names.
        """
        try:
            response = requests.get(f"{self.inference_base_url}/models")
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching model list: {e}")
            return []

    def unload_all_loras(self):
        """
        Unload all LORA adapters from the VLLM server.
        This method is a placeholder and should be implemented based on the server's capabilities.
        """
        # This method is not implemented in the base class
        models = self._get_vllm_model_list()
        for model in models:
            model_name = model.get("id", "")
            if model_name:
                try:
                    result = self.unload_lora(model_name)
                    print(f"Unloaded LORA adapter {model_name}: {result}")
                except Exception as e:
                    continue  # Skip models that cannot be unloaded


class ArtVLLMClient(VllmClient):
    def __init__(
        self,
        model: TrainableModel,
        **kwargs: Any,
    ):
        self.model = model
        self.client = model.openai_client()
        self.inference_base_url = model.inference_base_url
        # self.port = port
        self.base_model = model.base_model
        self.inference_name = model.get_inference_name()

    def get_inference_name(self) -> str:
        return self.model.get_inference_name()

    def load_lora(self, lora_name: str, lora_path: str):
        return {"success": True, "status_code": "art"}

    def unload_lora(self, lora_name: str):
        return {"success": True, "status_code": "art"}

    def _get_vllm_model_list(self) -> list[str]:
        return []  # are manages their own things

    def unload_all_loras(self):
        """
        Unload all LORA adapters from the Art VLLM client.
        This method is a placeholder and should be implemented based on the server's capabilities.
        """
        # This method is not implemented in the base class
        pass


class VllmRouter:
    """
    A router class to manage multiple VLLMClient instances.
    It distributes the load by returning a different client each time `get_client` is called.
    the class can be used as in iterator to get clients in a round-robin fashion.
    """

    def __init__(self, vllm_clients: list[VllmClient] = []):
        self.vllm_clients: list[VllmClient] = vllm_clients
        self.num_clients: KeyboardInterrupt = len(vllm_clients)

    def add_client(self, client: VllmClient):
        """
        Add a new VLLMClient to the router.
        """
        self.vllm_clients.append(client)
        self.num_clients = len(self.vllm_clients)
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        self.current_index = (self.current_index + 1) % self.num_clients
        client = self.vllm_clients[self.current_index]
        return client

    def get_client(self) -> VllmClient:
        """
        Get the current client in the round-robin rotation.
        """
        client = self.vllm_clients[self.current_index]
        return client

    def load_lora(self, lora_name: str, lora_path: str) -> bool:
        """
        Load a LORA adapter on all clients.
        """
        results = []
        for client in self.vllm_clients:
            result = client.load_lora(lora_name, lora_path)
            results.append(result)
        return all(result["success"] for result in results)

    def unload_lora(self, lora_name: str) -> bool:
        """
        Unload a LORA adapter on all clients.
        """
        results = []
        for client in self.vllm_clients:
            result = client.unload_lora(lora_name)
            results.append(result)
        return all(result["success"] for result in results)

    def unload_all_loras(self) -> None:
        """
        Unload all LORA adapters from all clients.
        """
        for client in self.vllm_clients:
            client.unload_all_loras()
