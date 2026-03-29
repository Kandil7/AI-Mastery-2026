"""
Agent Protocols Module

Production-ready protocol implementations:
- Model Context Protocol (MCP)
- Agent-to-Agent (A2A) Protocol

Features:
- Message formatting
- Tool discovery
- Resource sharing
- Inter-agent communication
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of protocol messages."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"


class ProtocolVersion(str, Enum):
    """Protocol versions."""

    V1 = "1.0"
    V2 = "2.0"


@dataclass
class Message:
    """Base protocol message."""

    id: str
    type: MessageType
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    version: str = ProtocolVersion.V1.value

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "jsonrpc": "2.0",
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "version": self.version,
        }

        if self.method:
            data["method"] = self.method
        if self.params:
            data["params"] = self.params
        if self.result is not None:
            data["result"] = self.result
        if self.error:
            data["error"] = self.error

        return data

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize from dict."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "request")),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
            timestamp=data.get("timestamp", time.time()),
            version=data.get("version", ProtocolVersion.V1.value),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def request(
        cls,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Message":
        """Create a request message."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            method=method,
            params=params,
        )

    @classmethod
    def response(
        cls,
        request_id: str,
        result: Any,
    ) -> "Message":
        """Create a response message."""
        return cls(
            id=request_id,
            type=MessageType.RESPONSE,
            result=result,
        )

    @classmethod
    def error(
        cls,
        request_id: str,
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> "Message":
        """Create an error message."""
        return cls(
            id=request_id,
            type=MessageType.ERROR,
            error={
                "code": code,
                "message": message,
                "data": data,
            },
        )


@dataclass
class ToolCall:
    """Tool call in protocol format."""

    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class ToolDefinition:
    """Tool definition for protocol."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
        }


@dataclass
class Resource:
    """Shared resource in protocol."""

    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    content: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
            "content": self.content,
        }


class ProtocolHandler(ABC):
    """Abstract base class for protocol handlers."""

    def __init__(self, version: str = ProtocolVersion.V1.value) -> None:
        self.version = version
        self._message_handlers: Dict[str, Callable] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}

    @abstractmethod
    async def send(self, message: Message) -> None:
        """Send a message."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncGenerator[Message, None]:
        """Receive messages."""
        pass

    def register_handler(
        self,
        method: str,
        handler: Callable[[Message], Any],
    ) -> None:
        """Register a message handler."""
        self._message_handlers[method] = handler
        logger.info(f"Registered handler for method: {method}")

    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message."""
        if message.type == MessageType.REQUEST:
            handler = self._message_handlers.get(message.method)
            if handler:
                try:
                    result = await handler(message)
                    return Message.response(message.id, result)
                except Exception as e:
                    return Message.error(
                        message.id,
                        code=-32000,
                        message=str(e),
                    )
            else:
                return Message.error(
                    message.id,
                    code=-32601,
                    message=f"Method not found: {message.method}",
                )

        elif message.type == MessageType.RESPONSE:
            # Resolve pending request
            if message.id in self._pending_requests:
                self._pending_requests[message.id].set_result(message)

        return None

    async def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Any:
        """Send request and wait for response."""
        message = Message.request(method, params)

        future = asyncio.Future()
        self._pending_requests[message.id] = future

        await self.send(message)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response.result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out: {method}")
        finally:
            del self._pending_requests[message.id]


class ModelContextProtocol(ProtocolHandler):
    """
    Model Context Protocol (MCP) implementation.

    Provides standardized interface for:
    - Tool discovery and execution
    - Resource access
    - Prompt templates
    """

    # MCP Methods
    METHOD_INITIALIZE = "initialize"
    METHOD_TOOLS_LIST = "tools/list"
    METHOD_TOOLS_CALL = "tools/call"
    METHOD_RESOURCES_LIST = "resources/list"
    METHOD_RESOURCES_READ = "resources/read"
    METHOD_PROMPTS_LIST = "prompts/list"
    METHOD_PROMPTS_GET = "prompts/get"

    def __init__(
        self,
        transport: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.transport = transport
        self._tools: Dict[str, ToolDefinition] = {}
        self._tool_handlers: Dict[str, Callable] = {}
        self._resources: Dict[str, Resource] = {}
        self._prompts: Dict[str, Dict[str, Any]] = {}

        self._initialize_handlers()

    def _initialize_handlers(self) -> None:
        """Initialize default MCP handlers."""
        self.register_handler(self.METHOD_TOOLS_LIST, self._handle_tools_list)
        self.register_handler(self.METHOD_TOOLS_CALL, self._handle_tools_call)
        self.register_handler(self.METHOD_RESOURCES_LIST, self._handle_resources_list)
        self.register_handler(self.METHOD_RESOURCES_READ, self._handle_resources_read)
        self.register_handler(self.METHOD_PROMPTS_LIST, self._handle_prompts_list)
        self.register_handler(self.METHOD_PROMPTS_GET, self._handle_prompts_get)

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """Register a tool."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
        )
        self._tool_handlers[name] = handler
        logger.info(f"Registered MCP tool: {name}")

    def register_resource(self, resource: Resource) -> None:
        """Register a resource."""
        self._resources[resource.uri] = resource
        logger.info(f"Registered MCP resource: {resource.uri}")

    def register_prompt(
        self,
        name: str,
        description: str,
        template: str,
        arguments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Register a prompt template."""
        self._prompts[name] = {
            "name": name,
            "description": description,
            "template": template,
            "arguments": arguments or [],
        }
        logger.info(f"Registered MCP prompt: {name}")

    async def _handle_tools_list(self, message: Message) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": [tool.to_dict() for tool in self._tools.values()],
        }

    async def _handle_tools_call(self, message: Message) -> Dict[str, Any]:
        """Handle tools/call request."""
        params = message.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self._tool_handlers:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self._tool_handlers[tool_name]
        result = await handler(**arguments) if asyncio.iscoroutinefunction(handler) else handler(**arguments)

        return {
            "content": [{"type": "text", "text": str(result)}],
        }

    async def _handle_resources_list(self, message: Message) -> Dict[str, Any]:
        """Handle resources/list request."""
        return {
            "resources": [r.to_dict() for r in self._resources.values()],
        }

    async def _handle_resources_read(self, message: Message) -> Dict[str, Any]:
        """Handle resources/read request."""
        params = message.params or {}
        uri = params.get("uri")

        if uri not in self._resources:
            raise ValueError(f"Unknown resource: {uri}")

        resource = self._resources[uri]
        return {
            "contents": [resource.to_dict()],
        }

    async def _handle_prompts_list(self, message: Message) -> Dict[str, Any]:
        """Handle prompts/list request."""
        return {
            "prompts": list(self._prompts.values()),
        }

    async def _handle_prompts_get(self, message: Message) -> Dict[str, Any]:
        """Handle prompts/get request."""
        params = message.params or {}
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name not in self._prompts:
            raise ValueError(f"Unknown prompt: {name}")

        prompt = self._prompts[name]
        template = prompt["template"]

        # Substitute arguments
        for key, value in arguments.items():
            template = template.replace(f"{{{key}}}", str(value))

        return {
            "description": prompt["description"],
            "messages": [{"role": "user", "content": template}],
        }

    async def send(self, message: Message) -> None:
        """Send message via transport."""
        if self.transport:
            await self.transport.send(message.to_json())
        else:
            logger.debug(f"MCP send: {message.to_json()}")

    async def receive(self) -> AsyncGenerator[Message, None]:
        """Receive messages from transport."""
        if self.transport:
            async for data in self.transport.receive():
                yield Message.from_json(data)

    async def initialize(self, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize MCP connection."""
        response = await self.request(
            self.METHOD_INITIALIZE,
            params={
                "protocolVersion": self.version,
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
                "clientInfo": client_info,
            },
        )
        return response or {}


class Agent2AgentProtocol(ProtocolHandler):
    """
    Agent-to-Agent (A2A) Protocol implementation.

    Enables communication between multiple agents:
    - Task delegation
    - Result sharing
    - Coordination
    """

    # A2A Methods
    METHOD_REGISTER = "a2a/register"
    METHOD_DEREGISTER = "a2a/deregister"
    METHOD_DISCOVER = "a2a/discover"
    METHOD_DELEGATE = "a2a/delegate"
    METHOD_BROADCAST = "a2a/broadcast"
    METHOD_QUERY = "a2a/query"

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        capabilities: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.agent_id = agent_id
        self.agent_name = agent_name
        self.capabilities = capabilities or {}

        self._registered_agents: Dict[str, Dict[str, Any]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()

        self._initialize_handlers()

    def _initialize_handlers(self) -> None:
        """Initialize A2A handlers."""
        self.register_handler(self.METHOD_REGISTER, self._handle_register)
        self.register_handler(self.METHOD_DEREGISTER, self._handle_deregister)
        self.register_handler(self.METHOD_DISCOVER, self._handle_discover)
        self.register_handler(self.METHOD_DELEGATE, self._handle_delegate)
        self.register_handler(self.METHOD_BROADCAST, self._handle_broadcast)
        self.register_handler(self.METHOD_QUERY, self._handle_query)

    async def _handle_register(self, message: Message) -> Dict[str, Any]:
        """Handle agent registration."""
        params = message.params or {}
        agent_info = {
            "id": params.get("id"),
            "name": params.get("name"),
            "capabilities": params.get("capabilities", {}),
            "status": "active",
            "registered_at": time.time(),
        }

        self._registered_agents[agent_info["id"]] = agent_info

        return {"success": True, "agent_id": agent_info["id"]}

    async def _handle_deregister(self, message: Message) -> Dict[str, Any]:
        """Handle agent deregistration."""
        params = message.params or {}
        agent_id = params.get("id")

        if agent_id in self._registered_agents:
            del self._registered_agents[agent_id]
            return {"success": True}

        return {"success": False, "error": "Agent not found"}

    async def _handle_discover(self, message: Message) -> Dict[str, Any]:
        """Handle agent discovery."""
        params = message.params or {}
        capability = params.get("capability")

        agents = list(self._registered_agents.values())

        if capability:
            agents = [
                a for a in agents
                if capability in a.get("capabilities", {})
            ]

        return {"agents": agents}

    async def _handle_delegate(self, message: Message) -> Dict[str, Any]:
        """Handle task delegation."""
        params = message.params or {}
        task = params.get("task")
        target_agent = params.get("target_agent")

        # Queue for processing
        await self._message_queue.put({
            "type": "delegation",
            "task": task,
            "target_agent": target_agent,
            "from_agent": self.agent_id,
        })

        return {"success": True, "queued": True}

    async def _handle_broadcast(self, message: Message) -> Dict[str, Any]:
        """Handle broadcast message."""
        params = message.params or {}
        content = params.get("content")

        await self._message_queue.put({
            "type": "broadcast",
            "content": content,
            "from_agent": self.agent_id,
        })

        return {"success": True, "recipients": len(self._registered_agents)}

    async def _handle_query(self, message: Message) -> Dict[str, Any]:
        """Handle agent query."""
        params = message.params or {}
        query = params.get("query")

        # Simple query handling
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "capabilities": self.capabilities,
            "status": "active",
        }

    async def register(self, transport: Optional[Any] = None) -> Dict[str, Any]:
        """Register this agent."""
        if transport:
            self.transport = transport

        return await self.request(
            self.METHOD_REGISTER,
            params={
                "id": self.agent_id,
                "name": self.agent_name,
                "capabilities": self.capabilities,
            },
        )

    async def discover_agents(
        self,
        capability: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Discover other agents."""
        response = await self.request(
            self.METHOD_DISCOVER,
            params={"capability": capability} if capability else None,
        )
        return response.get("agents", []) if response else []

    async def delegate_task(
        self,
        task: str,
        target_agent: str,
    ) -> Dict[str, Any]:
        """Delegate a task to another agent."""
        return await self.request(
            self.METHOD_DELEGATE,
            params={
                "task": task,
                "target_agent": target_agent,
            },
        )

    async def broadcast(self, content: str) -> Dict[str, Any]:
        """Broadcast message to all agents."""
        return await self.request(
            self.METHOD_BROADCAST,
            params={"content": content},
        )

    async def send(self, message: Message) -> None:
        """Send message."""
        if self.transport:
            await self.transport.send(message.to_json())
        else:
            logger.debug(f"A2A send: {message.to_json()}")

    async def receive(self) -> AsyncGenerator[Message, None]:
        """Receive messages."""
        if self.transport:
            async for data in self.transport.receive():
                yield Message.from_json(data)

    async def get_queued_messages(self) -> List[Dict[str, Any]]:
        """Get queued messages."""
        messages = []
        while not self._message_queue.empty():
            messages.append(await self._message_queue.get())
        return messages


class ProtocolRouter:
    """
    Routes messages to appropriate protocol handlers.

    Supports multiple protocols simultaneously.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, ProtocolHandler] = {}
        self._default_handler: Optional[str] = None

    def register_handler(
        self,
        protocol: str,
        handler: ProtocolHandler,
        default: bool = False,
    ) -> None:
        """Register a protocol handler."""
        self._handlers[protocol] = handler
        if default or not self._default_handler:
            self._default_handler = protocol
        logger.info(f"Registered protocol handler: {protocol}")

    def get_handler(self, protocol: Optional[str] = None) -> Optional[ProtocolHandler]:
        """Get handler for protocol."""
        protocol = protocol or self._default_handler
        return self._handlers.get(protocol)

    async def route(self, message: Message) -> Optional[Message]:
        """Route message to appropriate handler."""
        # Determine protocol from message
        protocol = self._detect_protocol(message)
        handler = self.get_handler(protocol)

        if handler:
            return await handler.handle_message(message)

        return Message.error(
            message.id,
            code=-32601,
            message=f"No handler for protocol: {protocol}",
        )

    def _detect_protocol(self, message: Message) -> Optional[str]:
        """Detect protocol from message."""
        method = message.method or ""

        if method.startswith("tools/") or method.startswith("resources/") or method.startswith("prompts/"):
            return "mcp"
        elif method.startswith("a2a/"):
            return "a2a"

        return self._default_handler


class InMemoryTransport:
    """In-memory transport for testing."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._connected: bool = False

    async def connect(self) -> None:
        """Connect transport."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect transport."""
        self._connected = False

    async def send(self, data: str) -> None:
        """Send data."""
        if self._connected:
            await self._queue.put(data)

    async def receive(self) -> AsyncGenerator[str, None]:
        """Receive data."""
        while self._connected:
            data = await self._queue.get()
            yield data
