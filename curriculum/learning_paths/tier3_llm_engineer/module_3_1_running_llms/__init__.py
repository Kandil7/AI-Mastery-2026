"""
Module 3.1: Running LLMs

Production-ready implementations for running Large Language Models:
- APIs: OpenAI, Anthropic, Google with retry logic and streaming
- Local Execution: llama.cpp, Ollama, LM Studio
- Prompt Engineering: Various prompting techniques
- Structured Output: JSON schema validation and constrained decoding
"""

from .apis import (
    OpenAIClient,
    AnthropicClient,
    GoogleClient,
    LLMProvider,
    create_client,
)
from .local_execution import (
    LlamaCppExecutor,
    OllamaExecutor,
    LMStudioExecutor,
    LocalModelConfig,
)
from .prompt_engineering import (
    PromptEngineer,
    PromptTemplate,
    FewShotExample,
    PromptStrategy,
)
from .structured_output import (
    StructuredOutputGenerator,
    JSONSchemaValidator,
    FunctionCaller,
    OutputSchema,
)

__all__ = [
    # APIs
    "OpenAIClient",
    "AnthropicClient",
    "GoogleClient",
    "LLMProvider",
    "create_client",
    # Local Execution
    "LlamaCppExecutor",
    "OllamaExecutor",
    "LMStudioExecutor",
    "LocalModelConfig",
    # Prompt Engineering
    "PromptEngineer",
    "PromptTemplate",
    "FewShotExample",
    "PromptStrategy",
    # Structured Output
    "StructuredOutputGenerator",
    "JSONSchemaValidator",
    "FunctionCaller",
    "OutputSchema",
]

__version__ = "1.0.0"
