import requests
from openai import AsyncOpenAI
from typing import Any, Dict, Optional
from openai.types.chat.chat_completion import ChatCompletion, Choice
from art.openai import patch_openai


class VLLMClient:
    def __init__(
        self,
        inference_base_url: str = "http://localhost",
        port: int = 8000,
        **kwargs: Any,
    ):
        self.inference_base_url = inference_base_url + f":{port}/v1"
        self.port = port
        self.client = AsyncOpenAI(
            base_url=self.inference_base_url,
            api_key="default",  # Assuming no API key is needed for local server
            **kwargs,  # Additional keyword arguments for the client
        )
        self.client = patch_openai(self.client)

    async def chat(
        self,
        message: Dict[str, str],
        model: str,
        **kwargs,
    ) -> dict:
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
