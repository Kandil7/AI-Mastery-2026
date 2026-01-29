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
        """
        ...

    def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> list[str]:  # Note: In real world use Iterator/Generator, using list[str] for simple typing in port
        """
        Generate text as a stream of tokens/chunks.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generator yielding text chunks
        """
        ...
