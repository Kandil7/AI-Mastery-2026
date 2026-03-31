"""
Ollama LLM Adapter
===================
Implementation of LLMPort for Ollama local LLM.

محول Ollama لنموذج اللغة المحلي
"""

import httpx

from src.domain.errors import LLMError


class OllamaLLM:
    """
    Ollama adapter implementing LLMPort.
    
    Ollama runs LLMs locally without API costs.
    
    محول Ollama لتشغيل نماذج اللغة محلياً
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 120.0,
    ) -> None:
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            model: Model name (llama3.1, mistral, etc.)
            timeout: Request timeout in seconds (LLMs can be slow)
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
    
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """
        Generate completion using Ollama API.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response (num_predict in Ollama)
            
        Returns:
            Generated text
            
        Raises:
            LLMError: On API errors
        """
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama HTTP error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise LLMError(f"Ollama connection error: {e}") from e
        except Exception as e:
            raise LLMError(f"Ollama error: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> any:  # Generator
        """
        Stream completion from Ollama API.
        
        توليد الرد بشكل متدفق من Ollama
        """
        import json
        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                            
        except Exception as e:
            raise LLMError(f"Ollama streaming error: {e}") from e


class OllamaChat:
    """
    Ollama chat adapter for multi-turn conversations.
    
    محول Ollama للمحادثات متعددة الدورات
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
    
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Assistant's response text
        """
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("message", {}).get("content", "")
            
        except Exception as e:
            raise LLMError(f"Ollama chat error: {e}") from e
