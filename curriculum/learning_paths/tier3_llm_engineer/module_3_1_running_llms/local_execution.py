"""
Local LLM Execution

Support for running LLMs locally using:
- llama.cpp: CPU/GPU inference with GGUF models
- Ollama: Simplified local model management
- LM Studio: Local API server with GUI

Features:
- Model loading and caching
- GPU acceleration support
- Quantization options
- Batch processing
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LocalModelConfig:
    """Configuration for local model execution."""

    model_path: str
    model_name: str
    n_ctx: int = 4096  # Context window
    n_gpu_layers: int = 0  # GPU layers (0 = CPU only)
    n_threads: int = 0  # CPU threads (0 = auto)
    n_batch: int = 512  # Batch size
    flash_attn: bool = False  # Flash attention
    mmap: bool = True  # Memory map
    mlock: bool = False  # Lock memory
    vocab_only: bool = False
    use_mmap: bool = True
    embedding: bool = False
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0

    # Quantization
    quantization: Optional[str] = None  # e.g., "Q4_K_M", "Q8_0"

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080

    def __post_init__(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning(f"Model path does not exist: {self.model_path}")


@dataclass
class LocalGenerationConfig:
    """Configuration for local generation."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    mirostat: int = 0  # 0 = disabled, 1 = v1, 2 = v2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    seed: int = -1  # -1 = random
    n_predict: int = 512
    stop: Optional[List[str]] = None
    stream: bool = False


class BaseLocalExecutor(ABC):
    """Abstract base class for local LLM executors."""

    def __init__(
        self,
        config: LocalModelConfig,
        generation_config: Optional[LocalGenerationConfig] = None,
    ) -> None:
        self.config = config
        self.generation_config = generation_config or LocalGenerationConfig()
        self._is_loaded = False
        self._process: Optional[subprocess.Popen] = None

    @abstractmethod
    async def load(self) -> bool:
        """Load the model into memory."""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LocalGenerationConfig] = None,
    ) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LocalGenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generation tokens."""
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded


class LlamaCppExecutor(BaseLocalExecutor):
    """
    llama.cpp executor for running GGUF models locally.

    Supports:
    - CPU and GPU inference
    - Multiple quantization formats
    - Flash attention
    - Memory mapping
    """

    def __init__(
        self,
        config: LocalModelConfig,
        generation_config: Optional[LocalGenerationConfig] = None,
        llama_cpp_path: Optional[str] = None,
    ) -> None:
        super().__init__(config, generation_config)
        self.llama_cpp_path = llama_cpp_path or self._find_llama_cpp()
        self._model = None

        # Try to import llama-cpp-python if available
        try:
            import llama_cpp
            self._llama_cpp = llama_cpp
            logger.info("llama-cpp-python imported successfully")
        except ImportError:
            self._llama_cpp = None
            logger.warning("llama-cpp-python not installed. Using CLI mode only.")

    def _find_llama_cpp(self) -> str:
        """Find llama.cpp installation."""
        possible_paths = [
            "llama-server",
            "llama-cli",
            "./llama.cpp/bin/llama-server",
            "./llama.cpp/llama-server",
            "/usr/local/bin/llama-server",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Check if in PATH
        import shutil
        server_path = shutil.which("llama-server")
        if server_path:
            return server_path

        return "llama-server"  # Default, may fail

    async def load(self) -> bool:
        """Load the model using llama-cpp-python."""
        if self._llama_cpp is None:
            logger.warning("Cannot load model: llama-cpp-python not installed")
            return False

        try:
            self._model = self._llama_cpp.Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads or None,
                n_batch=self.config.n_batch,
                flash_attn=self.config.flash_attn,
                use_mmap=self.config.mmap,
                use_mlock=self.config.mlock,
                vocab_only=self.config.vocab_only,
                embedding=self.config.embedding,
                verbose=True,
            )
            self._is_loaded = True
            logger.info(f"Loaded model: {self.config.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    async def unload(self) -> None:
        """Unload the model."""
        if self._model:
            del self._model
            self._model = None
            self._is_loaded = False
            logger.info("Model unloaded")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LocalGenerationConfig] = None,
    ) -> str:
        """Generate text using llama-cpp-python."""
        if not self._is_loaded:
            if not await self.load():
                raise RuntimeError("Model not loaded and failed to load")

        gen_config = config or self.generation_config

        # Format prompt with system message
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        try:
            output = self._model(
                full_prompt,
                max_tokens=gen_config.n_predict,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repeat_penalty=gen_config.repeat_penalty,
                frequency_penalty=gen_config.frequency_penalty,
                presence_penalty=gen_config.presence_penalty,
                stop=gen_config.stop or ["User:", "\n\n"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LocalGenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generation using llama-cpp-python."""
        if not self._is_loaded:
            if not await self.load():
                raise RuntimeError("Model not loaded and failed to load")

        gen_config = config or self.generation_config

        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        try:
            for token in self._model(
                full_prompt,
                max_tokens=gen_config.n_predict,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repeat_penalty=gen_config.repeat_penalty,
                stop=gen_config.stop or ["User:", "\n\n"],
                stream=True,
                echo=False,
            ):
                yield token["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self._model:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_path": self.config.model_path,
            "n_ctx": self.config.n_ctx,
            "n_gpu_layers": self.config.n_gpu_layers,
            "context_window": self._model.n_ctx(),
            "embedding_size": self._model.n_embd() if hasattr(self._model, "n_embd") else None,
        }


class OllamaExecutor(BaseLocalExecutor):
    """
    Ollama executor for running models via Ollama API.

    Ollama provides a simple way to run LLMs locally with
    automatic model downloading and management.
    """

    def __init__(
        self,
        config: LocalModelConfig,
        generation_config: Optional[LocalGenerationConfig] = None,
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__(config, generation_config)
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=300.0)
        self._model_name = config.model_name

    async def load(self) -> bool:
        """Check if model is available and pull if needed."""
        try:
            # Check if model exists
            response = await self._client.post(
                f"{self.base_url}/api/tags"
            )
            response.raise_for_status()
            models = response.json().get("models", [])

            model_exists = any(m["name"].startswith(self._model_name) for m in models)

            if not model_exists:
                logger.info(f"Model {self._model_name} not found. Pulling...")
                await self._pull_model()

            self._is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load/check model: {e}")
            return False

    async def _pull_model(self) -> None:
        """Pull model from Ollama registry."""
        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"name": self._model_name},
            timeout=None,
        ) as response:
            async for line in response.aiter_lines():
                try:
                    data = json.loads(line)
                    if "status" in data:
                        logger.info(f"Pull status: {data['status']}")
                except json.JSONDecodeError:
                    continue

    async def unload(self) -> None:
        """Ollama manages models automatically."""
        self._is_loaded = False
        logger.info("Ollama model marked as unloaded")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LocalGenerationConfig] = None,
    ) -> str:
        """Generate text using Ollama API."""
        gen_config = config or self.generation_config

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
                "repeat_penalty": gen_config.repeat_penalty,
                "num_predict": gen_config.n_predict,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if gen_config.stop:
            payload["options"]["stop"] = gen_config.stop

        response = await self._client.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LocalGenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generation using Ollama API."""
        gen_config = config or self.generation_config

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
                "repeat_penalty": gen_config.repeat_penalty,
                "num_predict": gen_config.n_predict,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if gen_config.stop:
            payload["options"]["stop"] = gen_config.stop

        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=None,
        ) as response:
            async for line in response.aiter_lines():
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information from Ollama."""
        try:
            response = await self._client.post(
                f"{self.base_url}/api/show",
                json={"name": self._model_name},
            )
            response.raise_for_status()
            info = response.json()

            return {
                "loaded": self._is_loaded,
                "model_name": self._model_name,
                "details": info.get("details", {}),
                "model_info": info.get("model_info", {}),
                "parameters": info.get("parameters", ""),
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"loaded": False, "error": str(e)}

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaExecutor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class LMStudioExecutor(BaseLocalExecutor):
    """
    LM Studio executor for running models via LM Studio API.

    LM Studio provides a user-friendly interface for
    downloading and running local LLMs with an OpenAI-compatible API.
    """

    def __init__(
        self,
        config: LocalModelConfig,
        generation_config: Optional[LocalGenerationConfig] = None,
        base_url: str = "http://localhost:1234",
    ) -> None:
        super().__init__(config, generation_config)
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=300.0)
        self._model_name = config.model_name

    async def load(self) -> bool:
        """Load model in LM Studio (model must be loaded via GUI or API)."""
        try:
            # Check if server is running
            response = await self._client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()

            models = response.json().get("data", [])
            model_loaded = any(
                m["id"] == self._model_name or self._model_name in m.get("name", "")
                for m in models
            )

            if not model_loaded:
                # Try to load the model
                await self._load_model()

            self._is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model in LM Studio: {e}")
            return False

    async def _load_model(self) -> None:
        """Load model via LM Studio API."""
        response = await self._client.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self._model_name,
                "prompt": "",
                "max_tokens": 1,
            },
        )
        if response.status_code != 200:
            logger.warning(f"Model may not be loaded in LM Studio: {response.status_code}")

    async def unload(self) -> None:
        """Unload model in LM Studio."""
        try:
            await self._client.post(f"{self.base_url}/v1/internal/unload")
            self._is_loaded = False
            logger.info("Model unloaded from LM Studio")
        except Exception as e:
            logger.warning(f"Failed to unload model: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LocalGenerationConfig] = None,
    ) -> str:
        """Generate text using LM Studio's OpenAI-compatible API."""
        gen_config = config or self.generation_config

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "max_tokens": gen_config.n_predict,
            "stream": False,
        }

        if gen_config.stop:
            payload["stop"] = gen_config.stop

        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LocalGenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generation using LM Studio API."""
        gen_config = config or self.generation_config

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "max_tokens": gen_config.n_predict,
            "stream": True,
        }

        if gen_config.stop:
            payload["stop"] = gen_config.stop

        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=None,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information from LM Studio."""
        try:
            response = await self._client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()

            models = response.json().get("data", [])
            model_info = next(
                (m for m in models if m["id"] == self._model_name),
                None,
            )

            return {
                "loaded": self._is_loaded,
                "model": model_info or {"id": self._model_name},
                "base_url": self.base_url,
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"loaded": False, "error": str(e)}

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "LMStudioExecutor":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class LocalModelManager:
    """
    Manager for local LLM models.

    Handles:
    - Model loading/unloading
    - Memory management
    - Model switching
    - Health checks
    """

    def __init__(self) -> None:
        self._executors: Dict[str, BaseLocalExecutor] = {}
        self._current_model: Optional[str] = None
        self._lock = asyncio.Lock()

    async def register_model(
        self,
        name: str,
        executor: BaseLocalExecutor,
    ) -> None:
        """Register a model executor."""
        async with self._lock:
            self._executors[name] = executor
            logger.info(f"Registered model: {name}")

    async def load_model(self, name: str) -> bool:
        """Load a registered model."""
        if name not in self._executors:
            logger.error(f"Model not registered: {name}")
            return False

        async with self._lock:
            # Unload current model if different
            if self._current_model and self._current_model != name:
                await self.unload_model(self._current_model)

            executor = self._executors[name]
            success = await executor.load()

            if success:
                self._current_model = name
                logger.info(f"Loaded model: {name}")

            return success

    async def unload_model(self, name: str) -> None:
        """Unload a model."""
        if name in self._executors:
            await self._executors[name].unload()
            if self._current_model == name:
                self._current_model = None
            logger.info(f"Unloaded model: {name}")

    async def generate(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate using a specific model."""
        if model_name not in self._executors:
            raise ValueError(f"Model not registered: {model_name}")

        executor = self._executors[model_name]
        if not executor.is_loaded:
            if not await self.load_model(model_name):
                raise RuntimeError(f"Failed to load model: {model_name}")

        return await executor.generate(prompt, system_prompt, **kwargs)

    async def generate_stream(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream generation using a specific model."""
        if model_name not in self._executors:
            raise ValueError(f"Model not registered: {model_name}")

        executor = self._executors[model_name]
        if not executor.is_loaded:
            if not await self.load_model(model_name):
                raise RuntimeError(f"Failed to load model: {model_name}")

        async for token in executor.generate_stream(prompt, system_prompt, **kwargs):
            yield token

    def get_registered_models(self) -> List[str]:
        """Get list of registered model names."""
        return list(self._executors.keys())

    def get_current_model(self) -> Optional[str]:
        """Get currently loaded model name."""
        return self._current_model

    async def health_check(self, model_name: str) -> Dict[str, Any]:
        """Check model health status."""
        if model_name not in self._executors:
            return {"status": "not_found", "error": "Model not registered"}

        executor = self._executors[model_name]
        info = await executor.get_model_info()
        info["status"] = "healthy" if executor.is_loaded else "not_loaded"
        return info

    async def shutdown(self) -> None:
        """Shutdown all models."""
        async with self._lock:
            for name, executor in self._executors.items():
                await executor.unload()
                if hasattr(executor, "close"):
                    await executor.close()
            self._executors.clear()
            self._current_model = None
            logger.info("All models shut down")
