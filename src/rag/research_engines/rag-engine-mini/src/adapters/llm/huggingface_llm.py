"""
Hugging Face LLM Adapter
========================
Implementation of the LLMPort using Hugging Face Inference API.

محول Hugging Face LLM - تنفيذ واجهة LLMPort باستخدام Hugging Face Inference API
"""

from typing import AsyncGenerator, Optional, List, Dict, Any
from huggingface_hub import AsyncInferenceClient

from src.application.ports.llm import LLMPort

class HuggingFaceLLM(LLMPort):
    """
    Adapter for Hugging Face LLM using Inference API.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.2,
        max_tokens: int = 700,
    ):
        self.client = AsyncInferenceClient(
            model=model_name,
            token=api_key
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response."""
        messages = self._build_messages(prompt, system_prompt, history)
        
        response = await self.client.chat_completion(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        messages = self._build_messages(prompt, system_prompt, history)
        
        async for chunk in await self.client.chat_completion(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            stream=True
        ):
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Helper to build message list."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
            
        messages.append({"role": "user", "content": prompt})
        return messages
