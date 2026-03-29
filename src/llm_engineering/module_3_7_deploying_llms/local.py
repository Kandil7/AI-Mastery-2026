"""
Local Deployment Module

Production-ready local deployment:
- Ollama deployment
- LM Studio deployment
- llama.cpp deployment
- Local API servers

Features:
- Easy setup
- Model management
- API compatibility
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LocalServerConfig:
    """Configuration for local server deployment."""

    # Model settings
    model_path: str
    model_name: str = ""
    model_url: Optional[str] = None  # For auto-download

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080
    workers: int = 1

    # Performance settings
    n_ctx: int = 4096
    n_gpu_layers: int = 0  # 0 = CPU only
    n_threads: int = 0  # 0 = auto
    n_batch: int = 512

    # Memory settings
    mmap: bool = True
    mlock: bool = False
    flash_attn: bool = False

    # API settings
    api_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = Path(self.model_path).stem


class LocalDeployment(ABC):
    """Abstract base class for local deployments."""

    def __init__(self, config: LocalServerConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._client = httpx.AsyncClient(
            base_url=f"http://{config.host}:{config.port}",
            timeout=120.0,
        )
        self._running = False

    @abstractmethod
    async def start(self) -> bool:
        """Start the deployment."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the deployment."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if deployment is healthy."""
        pass

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text from the deployed model."""
        if stream:
            return self._stream_generate(prompt, max_tokens, temperature)

        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Stream generation."""
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def list_models(self) -> List[str]:
        """List available models."""
        response = await self._client.get("/v1/models")
        response.raise_for_status()

        data = response.json()
        return [m["id"] for m in data.get("data", [])]

    async def close(self) -> None:
        """Close client."""
        await self._client.aclose()

    async def __aenter__(self) -> "LocalDeployment":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
        await self.close()


class OllamaDeployment(LocalDeployment):
    """
    Ollama deployment.

    Provides easy local model deployment with automatic
    model downloading and management.
    """

    def __init__(
        self,
        config: LocalServerConfig,
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        super().__init__(config)
        self.ollama_host = ollama_host
        self._ollama_client = httpx.AsyncClient(base_url=ollama_host, timeout=300.0)

    async def start(self) -> bool:
        """Start Ollama server."""
        # Check if Ollama is running
        try:
            response = await self._ollama_client.get("/api/tags")
            if response.status_code == 200:
                logger.info("Ollama server is already running")
                self._running = True
                return True
        except Exception:
            pass

        # Try to start Ollama
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            logger.error("Ollama not found in PATH")
            return False

        try:
            self._process = subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            await asyncio.sleep(3)

            # Verify
            if await self.health_check():
                self._running = True
                logger.info("Ollama server started")
                return True
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")

        return False

    async def stop(self) -> None:
        """Stop Ollama server."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
            self._running = False
            logger.info("Ollama server stopped")

    async def health_check(self) -> bool:
        """Check Ollama health."""
        try:
            response = await self._ollama_client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            async with self._ollama_client.stream(
                "POST",
                "/api/pull",
                json={"name": model_name},
            ) as response:
                async for line in response.aiter_lines():
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        logger.info(f"Pulling {model_name}: {status}")
                    except json.JSONDecodeError:
                        continue

            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List Ollama models."""
        response = await self._ollama_client.get("/api/tags")
        response.raise_for_status()

        data = response.json()
        return [m["name"] for m in data.get("models", [])]

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate using Ollama API."""
        if stream:
            return self._ollama_stream_generate(prompt, max_tokens, temperature)

        response = await self._ollama_client.post(
            "/api/generate",
            json={
                "model": self.config.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": False,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "")

    async def _ollama_stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Stream generation using Ollama."""
        async with self._ollama_client.stream(
            "POST",
            "/api/generate",
            json={
                "model": self.config.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": True,
            },
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

    async def close(self) -> None:
        """Close clients."""
        await self._ollama_client.aclose()
        await super().close()


class LMStudioDeployment(LocalDeployment):
    """
    LM Studio deployment.

    Uses LM Studio's local server with OpenAI-compatible API.
    """

    def __init__(
        self,
        config: LocalServerConfig,
        lmstudio_host: str = "http://localhost:1234",
    ) -> None:
        super().__init__(config)
        self.lmstudio_host = lmstudio_host
        self._client = httpx.AsyncClient(base_url=lmstudio_host, timeout=120.0)

    async def start(self) -> bool:
        """Start LM Studio server."""
        # LM Studio needs to be started manually via GUI
        # We just check if it's running

        if await self.health_check():
            self._running = True
            logger.info("LM Studio server is running")
            return True

        logger.warning("LM Studio server not running. Please start it from the GUI.")
        return False

    async def stop(self) -> None:
        """Stop LM Studio server."""
        # LM Studio doesn't have a clean shutdown API
        # User needs to close via GUI
        self._running = False

    async def health_check(self) -> bool:
        """Check LM Studio health."""
        try:
            response = await self._client.get("/v1/models")
            return response.status_code == 200
        except Exception:
            return False

    async def load_model(self, model_path: str) -> bool:
        """Load a model in LM Studio."""
        # LM Studio requires manual model loading via GUI
        # This is a placeholder for API-based loading if available
        logger.info(f"Please load model manually: {model_path}")
        return True

    async def list_models(self) -> List[str]:
        """List LM Studio models."""
        response = await self._client.get("/v1/models")
        response.raise_for_status()

        data = response.json()
        return [m["id"] for m in data.get("data", [])]


class LlamaCppDeployment(LocalDeployment):
    """
    llama.cpp deployment.

    Runs GGUF models using llama.cpp server.
    """

    def __init__(
        self,
        config: LocalServerConfig,
        llama_cpp_path: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self.llama_cpp_path = llama_cpp_path or self._find_llama_cpp()

    def _find_llama_cpp(self) -> str:
        """Find llama.cpp server binary."""
        possible_paths = [
            "llama-server",
            "./llama.cpp/llama-server",
            "./llama.cpp/bin/llama-server",
            "/usr/local/bin/llama-server",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Check PATH
        server_path = shutil.which("llama-server")
        if server_path:
            return server_path

        return "llama-server"

    async def start(self) -> bool:
        """Start llama.cpp server."""
        if not os.path.exists(self.config.model_path):
            logger.error(f"Model not found: {self.config.model_path}")
            return False

        # Build command
        cmd = [
            self.llama_cpp_path,
            "-m", self.config.model_path,
            "-h", self.config.host,
            "-p", str(self.config.port),
            "-c", str(self.config.n_ctx),
            "-b", str(self.config.n_batch),
        ]

        if self.config.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(self.config.n_gpu_layers)])

        if self.config.n_threads > 0:
            cmd.extend(["-t", str(self.config.n_threads)])

        if self.config.mlock:
            cmd.append("--mlock")

        if self.config.flash_attn:
            cmd.append("--flash-attn")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            await asyncio.sleep(5)

            if await self.health_check():
                self._running = True
                logger.info(f"llama.cpp server started on port {self.config.port}")
                return True
        except Exception as e:
            logger.error(f"Failed to start llama.cpp: {e}")

        return False

    async def stop(self) -> None:
        """Stop llama.cpp server."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
            self._running = False
            logger.info("llama.cpp server stopped")

    async def health_check(self) -> bool:
        """Check llama.cpp health."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            try:
                response = await self._client.get("/v1/models")
                return response.status_code == 200
            except Exception:
                return False


class LocalDeploymentManager:
    """
    Manager for local deployments.

    Provides unified interface for different local deployment options.
    """

    def __init__(self) -> None:
        self._deployments: Dict[str, LocalDeployment] = {}
        self._active: Optional[str] = None

    def create_deployment(
        self,
        name: str,
        deployment_type: str,
        config: LocalServerConfig,
        **kwargs: Any,
    ) -> LocalDeployment:
        """Create a deployment."""
        deployments = {
            "ollama": OllamaDeployment,
            "lmstudio": LMStudioDeployment,
            "llama_cpp": LlamaCppDeployment,
        }

        deployment_class = deployments.get(deployment_type.lower())
        if not deployment_class:
            raise ValueError(f"Unknown deployment type: {deployment_type}")

        deployment = deployment_class(config, **kwargs)
        self._deployments[name] = deployment

        return deployment

    async def start(self, name: str) -> bool:
        """Start a deployment."""
        if name not in self._deployments:
            logger.error(f"Deployment not found: {name}")
            return False

        deployment = self._deployments[name]
        success = await deployment.start()

        if success:
            self._active = name

        return success

    async def stop(self, name: str) -> None:
        """Stop a deployment."""
        if name in self._deployments:
            await self._deployments[name].stop()
            if self._active == name:
                self._active = None

    async def stop_all(self) -> None:
        """Stop all deployments."""
        for name in list(self._deployments.keys()):
            await self.stop(name)

    def get_active(self) -> Optional[LocalDeployment]:
        """Get active deployment."""
        if self._active and self._active in self._deployments:
            return self._deployments[self._active]
        return None

    def get(self, name: str) -> Optional[LocalDeployment]:
        """Get deployment by name."""
        return self._deployments.get(name)

    async def health_check(self, name: str) -> bool:
        """Check deployment health."""
        if name in self._deployments:
            return await self._deployments[name].health_check()
        return False

    def list_deployments(self) -> List[str]:
        """List deployment names."""
        return list(self._deployments.keys())

    async def __aenter__(self) -> "LocalDeploymentManager":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop_all()
