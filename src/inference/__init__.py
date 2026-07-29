# Backend-specific clients (e.g. `src.backends.verl.client.VerlClient`) are deliberately NOT
# exported here: this package must stay importable without any training framework installed.
from src.inference.client import InferenceClient, InferenceResponse
from src.inference.openai_client import OAIClient, OAIResponse


__all__ = [
    "InferenceClient",
    "InferenceResponse",
    "OAIClient",
    "OAIResponse",
]
