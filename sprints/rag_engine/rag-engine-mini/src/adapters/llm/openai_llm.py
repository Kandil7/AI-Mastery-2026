"""
OpenAI LLM Adapter
===================
Implementation of LLMPort for OpenAI API.

محول OpenAI لنموذج اللغة
"""

from openai import OpenAI

from src.domain.errors import LLMError, LLMRateLimitError


class OpenAILLM:
    """
    OpenAI ChatCompletion adapter implementing LLMPort.
    
    محول OpenAI ChatCompletion
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4o-mini, gpt-4o, etc.)
            timeout: Request timeout in seconds
        """
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
    
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """
        Generate completion using OpenAI Chat API.
        
        Args:
            prompt: The input prompt (or system+user combined)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            Generated text
            
        Raises:
            LLMError: On API errors
            LLMRateLimitError: On rate limit
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content
            return content or ""
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit
            if "rate_limit" in error_str or "429" in error_str:
                raise LLMRateLimitError() from e
            
            raise LLMError(f"OpenAI error: {e}") from e
