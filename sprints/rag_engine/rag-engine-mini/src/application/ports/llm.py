"""
LLM Port
=========
Interface for Large Language Model providers.

منفذ نموذج اللغة الكبير
"""

from typing import Protocol


class LLMPort(Protocol):
    """
    Port for LLM generation.
    
    Implementations: OpenAI, Ollama, Anthropic, etc.
    
    Design Decision: Using Protocol for structural subtyping.
    Any class with matching methods works without explicit inheritance.
    
    قرار التصميم: استخدام Protocol للتصنيف الهيكلي
    """
    
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """
        Generate text completion from prompt.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
            
        Raises:
            LLMError: If generation fails
        """
        ...
