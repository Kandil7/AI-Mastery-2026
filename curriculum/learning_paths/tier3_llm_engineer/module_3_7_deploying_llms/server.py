"""
LLM Server Module

Production-ready server deployments:
- vLLM server
- TGI (Text Generation Inference)
- FastAPI custom servers
- Load balancing

Features:
- High throughput
- Continuous batching
- Multi-model serving
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for LLM servers."""

    # Model settings
    model: str
    model_path: Optional[str] = None
    tokenizer: Optional[str] = None

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Performance settings
    max_batch_size: int = 256
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1

    # Quantization
    quantization: Optional[str] = None  # awq, gptq, squeezellm

    # API settings
    api_key: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

    # Logging
    log_level: str = "info"
    log_requests: bool = True


class LLMServer(ABC):
    """Abstract base class for LLM servers."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._client = httpx.AsyncClient(
            base_url=f"http://{config.host}:{config.port}",
            timeout=120.0,
        )
        self._running = False

    @abstractmethod
    async def start(self) -> bool:
        """Start the server."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the server."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check server health."""
        pass

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text."""
        if stream:
            return self._stream_generate(prompt, max_tokens, temperature)

        response = await self._client.post(
            "/v1/completions",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["text"]

    async def _stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Stream generation."""
        async with self._client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": self.config.model,
                "prompt": prompt,
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
                        content = chunk["choices"][0].get("text", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Chat completion."""
        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self.config.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def list_models(self) -> List[str]:
        """List available models."""
        response = await self._client.get("/v1/models")
        response.raise_for_status()

        data = response.json()
        return [m["id"] for m in data.get("data", [])]

    async def close(self) -> None:
        """Close client."""
        await self._client.aclose()

    async def __aenter__(self) -> "LLMServer":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
        await self.close()


class VLLMServer(LLMServer):
    """
    vLLM server deployment.

    High-throughput LLM serving with PagedAttention.
    """

    def __init__(
        self,
        config: ServerConfig,
        vllm_args: Optional[List[str]] = None,
    ) -> None:
        super().__init__(config)
        self.vllm_args = vllm_args or []

    async def start(self) -> bool:
        """Start vLLM server."""
        # Build command
        cmd = [
            "python", "-m", "vllm.entrypoints.api_server",
            "--model", self.config.model,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--max-model-len", str(self.config.max_model_len),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
        ]

        if self.config.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.config.tensor_parallel_size)])

        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])

        if self.config.tokenizer:
            cmd.extend(["--tokenizer", self.config.tokenizer])

        # Add custom args
        cmd.extend(self.vllm_args)

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            await asyncio.sleep(10)

            if await self.health_check():
                self._running = True
                logger.info(f"vLLM server started on port {self.config.port}")
                return True
        except Exception as e:
            logger.error(f"Failed to start vLLM: {e}")

        return False

    async def stop(self) -> None:
        """Stop vLLM server."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
            self._running = False
            logger.info("vLLM server stopped")

    async def health_check(self) -> bool:
        """Check vLLM health."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            try:
                response = await self._client.get("/v1/models")
                return response.status_code == 200
            except Exception:
                return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get vLLM statistics."""
        try:
            response = await self._client.get("/stats")
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}


class TGIServer(LLMServer):
    """
    Text Generation Inference (TGI) server.

    Hugging Face's optimized inference server.
    """

    def __init__(
        self,
        config: ServerConfig,
        docker_image: str = "ghcr.io/huggingface/text-generation-inference:latest",
    ) -> None:
        super().__init__(config)
        self.docker_image = docker_image

    async def start(self) -> bool:
        """Start TGI server using Docker."""
        # Check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Docker not available")
                return False
        except Exception:
            logger.error("Failed to check Docker")
            return False

        # Build Docker command
        cmd = [
            "docker", "run", "--rm",
            "-p", f"{self.config.port}:{self.config.port}",
            "-e", f"MODEL_ID={self.config.model}",
            "-e", f"PORT={self.config.port}",
            "-e", f"MAX_INPUT_LENGTH={self.config.max_model_len // 2}",
            "-e", f"MAX_TOTAL_TOKENS={self.config.max_model_len}",
            "-e", f"MAX_BATCH_SIZE={self.config.max_batch_size}",
        ]

        if self.config.model_path:
            cmd.extend(["-v", f"{self.config.model_path}:/data"])

        cmd.extend([
            self.docker_image,
            "--json-output",
        ])

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            await asyncio.sleep(15)

            if await self.health_check():
                self._running = True
                logger.info(f"TGI server started on port {self.config.port}")
                return True
        except Exception as e:
            logger.error(f"Failed to start TGI: {e}")

        return False

    async def stop(self) -> None:
        """Stop TGI server."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
            self._running = False
            logger.info("TGI server stopped")

    async def health_check(self) -> bool:
        """Check TGI health."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_info(self) -> Dict[str, Any]:
        """Get TGI server info."""
        try:
            response = await self._client.get("/info")
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}


class FastAPIServer(LLMServer):
    """
    Custom FastAPI LLM server.

    Flexible server with custom endpoints.
    """

    def __init__(
        self,
        config: ServerConfig,
        model_loader: Optional[Callable] = None,
        generate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(config)
        self.model_loader = model_loader
        self.generate_fn = generate_fn
        self._app = None
        self._model = None

    def create_app(self) -> Any:
        """Create FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
        except ImportError:
            raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

        app = FastAPI(
            title="LLM API",
            description="Custom LLM inference API",
            version="1.0.0",
        )

        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request/Response models
        class CompletionRequest(BaseModel):
            model: str
            prompt: str
            max_tokens: int = 256
            temperature: float = 0.7
            stream: bool = False

        class CompletionResponse(BaseModel):
            id: str
            model: str
            choices: List[Dict[str, Any]]
            usage: Dict[str, int]

        class ChatRequest(BaseModel):
            model: str
            messages: List[Dict[str, str]]
            max_tokens: int = 256
            temperature: float = 0.7

        # Endpoints
        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.get("/v1/models")
        async def list_models():
            return {
                "data": [{"id": self.config.model, "object": "model"}]
            }

        @app.post("/v1/completions", response_model=CompletionResponse)
        async def completions(request: CompletionRequest):
            if not self.generate_fn:
                raise HTTPException(500, "No generation function configured")

            result = await self.generate_fn(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            return CompletionResponse(
                id=f"cmpl-{int(time.time())}",
                model=request.model,
                choices=[{
                    "text": result,
                    "index": 0,
                    "finish_reason": "stop",
                }],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatRequest):
            if not self.generate_fn:
                raise HTTPException(500, "No generation function configured")

            # Convert messages to prompt
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in request.messages)

            result = await self.generate_fn(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            return {
                "id": f"chat-{int(time.time())}",
                "model": request.model,
                "choices": [{
                    "message": {"role": "assistant", "content": result},
                    "index": 0,
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        self._app = app
        return app

    async def start(self) -> bool:
        """Start FastAPI server."""
        if not self._app:
            self.create_app()

        try:
            import uvicorn

            # Run in thread
            loop = asyncio.get_event_loop()

            def run_server():
                uvicorn.run(
                    self._app,
                    host=self.config.host,
                    port=self.config.port,
                    log_level=self.config.log_level,
                )

            asyncio.create_task(loop.run_in_executor(None, run_server))

            # Wait for server to start
            await asyncio.sleep(3)

            if await self.health_check():
                self._running = True
                logger.info(f"FastAPI server started on port {self.config.port}")
                return True
        except Exception as e:
            logger.error(f"Failed to start FastAPI: {e}")

        return False

    async def stop(self) -> None:
        """Stop FastAPI server."""
        self._running = False
        logger.info("FastAPI server stopped")

    async def health_check(self) -> bool:
        """Check FastAPI health."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False


class LoadBalancer:
    """
    Load balancer for multiple LLM servers.

    Distributes requests across server instances.
    """

    def __init__(
        self,
        servers: Optional[List[str]] = None,
        strategy: str = "round_robin",
        health_check_interval: float = 30.0,
    ) -> None:
        self.servers = servers or []
        self.strategy = strategy
        self.health_check_interval = health_check_interval

        self._current_index = 0
        self._server_stats: Dict[str, Dict[str, Any]] = {}
        self._healthy_servers: Set[str] = set()

        self._running = False
        self._health_task: Optional[asyncio.Task] = None

    def add_server(self, url: str) -> None:
        """Add a server to the pool."""
        if url not in self.servers:
            self.servers.append(url)
            self._server_stats[url] = {
                "requests": 0,
                "errors": 0,
                "avg_latency": 0.0,
                "last_check": 0.0,
            }

    def remove_server(self, url: str) -> None:
        """Remove a server from the pool."""
        if url in self.servers:
            self.servers.remove(url)
            self._server_stats.pop(url, None)
            self._healthy_servers.discard(url)

    async def start(self) -> None:
        """Start load balancer."""
        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())

        # Initial health check
        await self._check_all_servers()

    async def stop(self) -> None:
        """Stop load balancer."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            await self._check_all_servers()
            await asyncio.sleep(self.health_check_interval)

    async def _check_all_servers(self) -> None:
        """Check health of all servers."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for url in self.servers:
                try:
                    response = await client.get(f"{url}/health")
                    if response.status_code == 200:
                        self._healthy_servers.add(url)
                    else:
                        self._healthy_servers.discard(url)
                except Exception:
                    self._healthy_servers.discard(url)

    def get_server(self) -> Optional[str]:
        """Get next server based on strategy."""
        healthy = list(self._healthy_servers)

        if not healthy:
            return None

        if self.strategy == "round_robin":
            server = healthy[self._current_index % len(healthy)]
            self._current_index += 1
            return server

        elif self.strategy == "least_requests":
            return min(
                healthy,
                key=lambda s: self._server_stats.get(s, {}).get("requests", 0),
            )

        elif self.strategy == "least_latency":
            return min(
                healthy,
                key=lambda s: self._server_stats.get(s, {}).get("avg_latency", float("inf")),
            )

        return healthy[0]

    async def forward_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        method: str = "POST",
    ) -> Any:
        """Forward request to a server."""
        server = self.get_server()

        if not server:
            raise RuntimeError("No healthy servers available")

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if method == "POST":
                    response = await client.post(f"{server}{endpoint}", json=data)
                else:
                    response = await client.get(f"{server}{endpoint}")

                response.raise_for_status()

                # Update stats
                latency = (time.time() - start_time) * 1000
                self._update_stats(server, success=True, latency=latency)

                return response.json()

        except Exception as e:
            self._update_stats(server, success=False, latency=0)
            raise

    def _update_stats(
        self,
        server: str,
        success: bool,
        latency: float,
    ) -> None:
        """Update server statistics."""
        if server not in self._server_stats:
            self._server_stats[server] = {
                "requests": 0,
                "errors": 0,
                "avg_latency": 0.0,
            }

        stats = self._server_stats[server]
        stats["requests"] += 1

        if not success:
            stats["errors"] += 1

        # Update average latency
        n = stats["requests"]
        stats["avg_latency"] = (stats["avg_latency"] * (n - 1) + latency) / n

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            "total_servers": len(self.servers),
            "healthy_servers": len(self._healthy_servers),
            "strategy": self.strategy,
            "server_stats": self._server_stats,
        }


# Import Set for type hints
from typing import Set
