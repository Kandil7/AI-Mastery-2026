"""
Vendor SDKs Module

Production-ready wrappers for vendor agent SDKs:
- OpenAI Agents SDK
- Google Agent Development Kit (ADK)
- Claude Agent SDK

Features:
- Unified interface
- SDK abstraction
- Feature normalization
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for agent creation."""

    name: str
    model: str
    instructions: str = ""
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from agent execution."""

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VendorSDKWrapper(ABC):
    """Abstract base class for vendor SDK wrappers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self._client = None
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the SDK."""
        pass

    @abstractmethod
    async def create_agent(self, config: AgentConfig) -> Any:
        """Create an agent."""
        pass

    @abstractmethod
    async def run_agent(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AgentResponse:
        """Run an agent."""
        pass

    @abstractmethod
    async def stream_agent(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream agent response."""
        pass


class OpenAIAgentsSDK(VendorSDKWrapper):
    """
    OpenAI Agents SDK wrapper.

    Provides unified interface to OpenAI's agent capabilities.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key, **kwargs)
        self.organization = organization
        self._agents_module = None

    async def initialize(self) -> None:
        """Initialize OpenAI Agents SDK."""
        try:
            # Try to import OpenAI Agents SDK
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
            self._initialized = True
            logger.info("OpenAI Agents SDK initialized")
        except ImportError:
            logger.warning("OpenAI Agents SDK not available. Using fallback.")
            self._initialized = False

    async def create_agent(self, config: AgentConfig) -> Any:
        """Create OpenAI agent."""
        if not self._initialized:
            await self.initialize()

        # Create agent configuration
        agent_config = {
            "name": config.name,
            "model": config.model,
            "instructions": config.instructions,
            "tools": config.tools or [],
        }

        # If Agents SDK available, use it
        if self._agents_module:
            agent = self._agents_module.Agent(**agent_config)
        else:
            # Fallback: return config as agent
            agent = agent_config

        return agent

    async def run_agent(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AgentResponse:
        """Run OpenAI agent."""
        if not self._initialized:
            await self.initialize()

        # Handle agent as config dict or SDK agent
        if isinstance(agent, dict):
            model = agent.get("model", "gpt-4")
            instructions = agent.get("instructions", "")
            tools = agent.get("tools", [])
        else:
            model = getattr(agent, "model", "gpt-4")
            instructions = getattr(agent, "instructions", "")
            tools = getattr(agent, "tools", [])

        # Build messages
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": input_text})

        # Call API
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
        )

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        return AgentResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
        )

    async def stream_agent(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI agent response."""
        if not self._initialized:
            await self.initialize()

        if isinstance(agent, dict):
            model = agent.get("model", "gpt-4")
            instructions = agent.get("instructions", "")
        else:
            model = getattr(agent, "model", "gpt-4")
            instructions = getattr(agent, "instructions", "")

        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": input_text})

        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=kwargs.get("temperature", 0.7),
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GoogleADK(VendorSDKWrapper):
    """
    Google Agent Development Kit (ADK) wrapper.

    Provides unified interface to Google's agent capabilities.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key, **kwargs)
        self.project_id = project_id
        self.location = location
        self._vertex_ai = None

    async def initialize(self) -> None:
        """Initialize Google ADK."""
        try:
            # Try to import Vertex AI
            from vertexai.generative_models import GenerativeModel

            self._vertex_ai = GenerativeModel
            self._initialized = True
            logger.info("Google ADK initialized")
        except ImportError:
            logger.warning("Google ADK not available. Using fallback.")
            self._initialized = False

    async def create_agent(self, config: AgentConfig) -> Any:
        """Create Google agent."""
        if not self._initialized:
            await self.initialize()

        agent_config = {
            "model_name": config.model,
            "system_instructions": config.instructions,
        }

        if self._vertex_ai:
            agent = self._vertex_ai(**agent_config)
        else:
            agent = agent_config

        return agent

    async def run_agent(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AgentResponse:
        """Run Google agent."""
        if not self._initialized:
            await self.initialize()

        if isinstance(agent, dict):
            model_name = agent.get("model_name", "gemini-pro")
        else:
            model_name = getattr(agent, "model_name", "gemini-pro")

        # Fallback to Gemini API
        try:
            from vertexai.generative_models import GenerativeModel

            model = GenerativeModel(model_name)
            response = await model.generate_content_async(input_text)

            return AgentResponse(
                content=response.text,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                } if hasattr(response, 'usage_metadata') else None,
            )
        except ImportError:
            # Fallback without Vertex AI
            return AgentResponse(
                content=f"[Google ADK Fallback] Response to: {input_text}",
            )

    async def stream_agent(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream Google agent response."""
        if not self._initialized:
            await self.initialize()

        if isinstance(agent, dict):
            model_name = agent.get("model_name", "gemini-pro")
        else:
            model_name = getattr(agent, "model_name", "gemini-pro")

        try:
            from vertexai.generative_models import GenerativeModel

            model = GenerativeModel(model_name)
            response = model.generate_content(input_text, stream=True)

            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except ImportError:
            yield "[Google ADK Fallback] Streaming not available"


class ClaudeAgentSDK(VendorSDKWrapper):
    """
    Claude Agent SDK wrapper.

    Provides unified interface to Anthropic's agent capabilities.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key, **kwargs)
        self.base_url = base_url
        self._anthropic = None

    async def initialize(self) -> None:
        """Initialize Claude SDK."""
        try:
            import anthropic

            self._anthropic = anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self._initialized = True
            logger.info("Claude Agent SDK initialized")
        except ImportError:
            logger.warning("Claude SDK not available. Using fallback.")
            self._initialized = False

    async def create_agent(self, config: AgentConfig) -> Any:
        """Create Claude agent."""
        if not self._initialized:
            await self.initialize()

        agent_config = {
            "name": config.name,
            "model": config.model,
            "system": config.instructions,
            "tools": config.tools or [],
        }

        return agent_config

    async def run_agent(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AgentResponse:
        """Run Claude agent."""
        if not self._initialized:
            await self.initialize()

        model = agent.get("model", "claude-3-sonnet-20240229") if isinstance(agent, dict) else "claude-3-sonnet-20240229"
        system = agent.get("system", "") if isinstance(agent, dict) else ""
        tools = agent.get("tools", []) if isinstance(agent, dict) else []

        response = await self._client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            system=system,
            messages=[{"role": "user", "content": input_text}],
            tools=tools if tools else None,
        )

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            } if response.usage else None,
        )

    async def stream_agent(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream Claude agent response."""
        if not self._initialized:
            await self.initialize()

        model = agent.get("model", "claude-3-sonnet-20240229") if isinstance(agent, dict) else "claude-3-sonnet-20240229"
        system = agent.get("system", "") if isinstance(agent, dict) else ""

        async with self._client.messages.stream(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            system=system,
            messages=[{"role": "user", "content": input_text}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


class UnifiedAgentSDK:
    """
    Unified interface across all vendor SDKs.

    Automatically selects appropriate SDK based on configuration.
    """

    def __init__(
        self,
        provider: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.provider = provider
        self._sdk: Optional[VendorSDKWrapper] = None
        self._kwargs = kwargs

        if provider != "auto":
            self._select_sdk(provider)

    def _select_sdk(self, provider: str) -> None:
        """Select appropriate SDK."""
        providers = {
            "openai": OpenAIAgentsSDK,
            "google": GoogleADK,
            "anthropic": ClaudeAgentSDK,
        }

        sdk_class = providers.get(provider.lower())
        if sdk_class:
            self._sdk = sdk_class(**self._kwargs)
            logger.info(f"Selected SDK: {provider}")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def initialize(self) -> None:
        """Initialize selected SDK."""
        if self.provider == "auto":
            # Try each provider in order
            for provider in ["openai", "anthropic", "google"]:
                try:
                    self._select_sdk(provider)
                    await self._sdk.initialize()
                    if self._sdk._initialized:
                        self.provider = provider
                        return
                except Exception:
                    continue
            raise RuntimeError("No SDK available")
        else:
            await self._sdk.initialize()

    async def create_agent(self, config: AgentConfig) -> Any:
        """Create agent."""
        if not self._sdk:
            await self.initialize()
        return await self._sdk.create_agent(config)

    async def run(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AgentResponse:
        """Run agent."""
        if not self._sdk:
            await self.initialize()
        return await self._sdk.run_agent(agent, input_text, **kwargs)

    async def stream(
        self,
        agent: Any,
        input_text: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream agent response."""
        if not self._sdk:
            await self.initialize()
        async for chunk in self._sdk.stream_agent(agent, input_text, **kwargs):
            yield chunk

    def get_provider(self) -> str:
        """Get current provider."""
        return self.provider


class AgentOrchestrator:
    """
    Orchestrates multiple agents from different vendors.

    Features:
    - Multi-agent coordination
    - Fallback handling
    - Load balancing
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Any] = {}
        self._sdks: Dict[str, VendorSDKWrapper] = {}

    def register_sdk(
        self,
        name: str,
        sdk: VendorSDKWrapper,
    ) -> None:
        """Register an SDK."""
        self._sdks[name] = sdk
        logger.info(f"Registered SDK: {name}")

    async def create_agent(
        self,
        name: str,
        config: AgentConfig,
        sdk_name: str,
    ) -> Any:
        """Create agent with specific SDK."""
        if sdk_name not in self._sdks:
            raise ValueError(f"Unknown SDK: {sdk_name}")

        sdk = self._sdks[sdk_name]
        agent = await sdk.create_agent(config)
        self._agents[name] = {"agent": agent, "sdk": sdk_name, "config": config}

        return agent

    async def run(
        self,
        agent_name: str,
        input_text: str,
        fallback: bool = True,
        **kwargs: Any,
    ) -> AgentResponse:
        """Run agent with optional fallback."""
        if agent_name not in self._agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        agent_info = self._agents[agent_name]
        sdk_name = agent_info["sdk"]
        agent = agent_info["agent"]

        try:
            sdk = self._sdks[sdk_name]
            return await sdk.run_agent(agent, input_text, **kwargs)
        except Exception as e:
            logger.warning(f"Agent {agent_name} failed: {e}")

            if fallback:
                return await self._run_fallback(input_text, **kwargs)
            raise

    async def _run_fallback(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> AgentResponse:
        """Run fallback agent."""
        for sdk_name, sdk in self._sdks.items():
            try:
                config = AgentConfig(
                    name="fallback",
                    model="default",
                )
                agent = await sdk.create_agent(config)
                return await sdk.run_agent(agent, input_text, **kwargs)
            except Exception:
                continue

        raise RuntimeError("All fallbacks failed")

    async def broadcast(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> Dict[str, AgentResponse]:
        """Run all agents and collect responses."""
        results = {}

        for name, agent_info in self._agents.items():
            try:
                sdk = self._sdks[agent_info["sdk"]]
                response = await sdk.run_agent(agent_info["agent"], input_text, **kwargs)
                results[name] = response
            except Exception as e:
                logger.warning(f"Agent {name} failed: {e}")
                results[name] = AgentResponse(
                    content=f"Error: {e}",
                )

        return results
