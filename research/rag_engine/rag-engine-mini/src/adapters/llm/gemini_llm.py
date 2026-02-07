"""
Gemini LLM Adapter
==================
Implementation of the LLMPort using Google's Gemini models.

محول Gemini LLM - تنفيذ واجهة LLMPort باستخدام نماذج Google Gemini
"""

import google.generativeai as genai
from typing import AsyncGenerator, Optional, List, Dict, Any

from src.application.ports.llm import LLMPort

class GeminiLLM(LLMPort):
    """
    Adapter for Google Gemini LLM.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 700,
    ):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = genai.GenerativeModel(model_name)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response."""
        # Note: Gemini handle system instruction in GenModel init or as first turn
        # Here we simplify for the port interface
        full_context = ""
        if system_prompt:
            full_context += f"System: {system_prompt}\n\n"
        
        if history:
            for turn in history:
                role = "User" if turn["role"] == "user" else "Model"
                full_context += f"{role}: {turn['content']}\n"
        
        full_context += f"User: {prompt}"
        
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        response = await self.model.generate_content_async(
            full_context,
            generation_config=generation_config
        )
        return response.text

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        full_context = ""
        if system_prompt:
            full_context += f"System: {system_prompt}\n\n"
        
        if history:
            for turn in history:
                role = "User" if turn["role"] == "user" else "Model"
                full_context += f"{role}: {turn['content']}\n"
        
        full_context += f"User: {prompt}"
        
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        response = await self.model.generate_content_async(
            full_context,
            generation_config=generation_config,
            stream=True
        )
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text
