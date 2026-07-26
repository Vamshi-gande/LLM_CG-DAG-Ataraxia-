from typing import Any, Dict

import httpx


class OllamaClient:
    """
    Async HTTP client for forwarding requests to the Ollama backend.
    Wraps httpx.AsyncClient with a configurable base URL.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        POST /api/chat with modified messages.
        Returns parsed JSON response dict.
        """
        payload = {"model": model, "messages": messages, "stream": stream, **kwargs}
        response = await self._client.post(
            f"{self._base_url}/api/chat", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """POST /api/generate with modified prompt."""
        payload = {"model": model, "prompt": prompt, "stream": stream, **kwargs}
        response = await self._client.post(
            f"{self._base_url}/api/generate", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()