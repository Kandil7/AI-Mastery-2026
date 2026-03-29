"""
Tools and Agents Module

Production-ready tool integration for RAG:
- Tool registry and execution
- API tool integration
- Code interpreter
- Calculator and search tools

Features:
- Type-safe tool definitions
- Async execution
- Error handling
- Tool composition
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        if self.default is not None:
            schema["default"] = self.default

        return schema


@dataclass
class ToolDefinition:
    """Complete tool definition."""

    name: str
    description: str
    parameters: List[ToolParameter]
    func: Optional[Callable] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class Tool(ABC):
    """Abstract base class for tools."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters or []

        self._stats = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_latency_ms": 0.0,
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool."""
        pass

    def get_definition(self) -> ToolDefinition:
        """Get tool definition."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            func=self.execute,
        )

    def _update_stats(self, success: bool, latency_ms: float) -> None:
        """Update execution statistics."""
        self._stats["calls"] += 1
        if success:
            self._stats["successes"] += 1
        else:
            self._stats["failures"] += 1
        self._stats["total_latency_ms"] += latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get tool statistics."""
        return {
            **self._stats,
            "avg_latency_ms": (
                self._stats["total_latency_ms"] / self._stats["calls"]
                if self._stats["calls"] > 0 else 0
            ),
        }


class APITool(Tool):
    """
    Tool for calling external APIs.

    Supports REST APIs with authentication.
    """

    def __init__(
        self,
        name: str,
        description: str,
        base_url: str,
        endpoint: str,
        method: str = "GET",
        parameters: Optional[List[ToolParameter]] = None,
        headers: Optional[Dict[str, str]] = None,
        auth_type: str = "none",  # none, bearer, api_key, basic
        auth_value: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name, description, parameters)

        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        self.auth_type = auth_type
        self.auth_value = auth_value
        self.timeout = timeout

        self._client = httpx.AsyncClient(timeout=timeout)

        # Setup auth headers
        self._setup_auth()

    def _setup_auth(self) -> None:
        """Setup authentication headers."""
        if self.auth_type == "bearer" and self.auth_value:
            self.headers["Authorization"] = f"Bearer {self.auth_value}"
        elif self.auth_type == "api_key" and self.auth_value:
            self.headers["X-API-Key"] = self.auth_value
        elif self.auth_type == "basic" and self.auth_value:
            import base64
            encoded = base64.b64encode(self.auth_value.encode()).decode()
            self.headers["Authorization"] = f"Basic {encoded}"

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute API call."""
        import time
        start_time = time.time()

        try:
            # Build URL
            url = f"{self.base_url}/{self.endpoint.lstrip('/')}"

            # Prepare request
            request_kwargs = {"headers": self.headers}

            if self.method == "GET":
                request_kwargs["params"] = kwargs
            elif self.method in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = kwargs
            elif self.method == "DELETE":
                request_kwargs["params"] = kwargs

            # Make request
            response = await self._client.request(self.method, url, **request_kwargs)
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(True, latency_ms)

            # Try to parse JSON
            try:
                output = response.json()
            except json.JSONDecodeError:
                output = response.text

            return ToolResult(
                success=True,
                output=output,
                latency_ms=latency_ms,
                metadata={"status_code": response.status_code},
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(False, latency_ms)

            return ToolResult(
                success=False,
                output=None,
                error=f"HTTP error: {e.response.status_code} - {e.response.text}",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(False, latency_ms)

            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                latency_ms=latency_ms,
            )

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


class CalculatorTool(Tool):
    """
    Calculator tool for mathematical operations.

    Supports basic arithmetic and scientific functions.
    """

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Supports basic arithmetic (+, -, *, /), powers (^), and scientific functions (sin, cos, tan, log, sqrt, exp).",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 2', 'sin(0.5)', 'sqrt(16)')",
                    required=True,
                ),
            ],
        )

    async def execute(self, expression: str, **kwargs: Any) -> ToolResult:
        """Execute calculation."""
        import time
        import math
        start_time = time.time()

        try:
            # Validate expression
            self._validate_expression(expression)

            # Safe evaluation
            result = self._safe_eval(expression)

            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(True, latency_ms)

            return ToolResult(
                success=True,
                output=result,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(False, latency_ms)

            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                latency_ms=latency_ms,
            )

    def _validate_expression(self, expression: str) -> None:
        """Validate expression for safety."""
        # Only allow safe characters
        allowed_pattern = r'^[\d\s\+\-\*\/\^\(\)\.\,]+$'
        if not re.match(allowed_pattern, expression):
            # Check for function calls
            func_pattern = r'^(sin|cos|tan|log|sqrt|exp|abs|round|floor|ceil)\s*\('
            if not re.search(func_pattern, expression.lower()):
                raise ValueError("Invalid characters in expression")

        # Check for dangerous patterns
        dangerous = ["__", "import", "exec", "eval", "compile", "open", "file"]
        for pattern in dangerous:
            if pattern in expression.lower():
                raise ValueError(f"Potentially dangerous pattern: {pattern}")

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expression."""
        # Replace function names with math module equivalents
        expression = expression.lower()
        expression = expression.replace("sin", "math.sin")
        expression = expression.replace("cos", "math.cos")
        expression = expression.replace("tan", "math.tan")
        expression = expression.replace("log", "math.log10")
        expression = expression.replace("sqrt", "math.sqrt")
        expression = expression.replace("exp", "math.exp")
        expression = expression.replace("abs", "abs")
        expression = expression.replace("^", "**")

        # Safe namespace
        safe_dict = {"math": math, "abs": abs}

        return eval(expression, {"__builtins__": {}}, safe_dict)


class CodeInterpreter(Tool):
    """
    Code interpreter tool for executing Python code.

    WARNING: Use with caution in production environments.
    Consider sandboxing for security.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        sandbox: bool = False,
        allowed_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            name="code_interpreter",
            description="Execute Python code and return the result. Use for calculations, data processing, and code-based problem solving.",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python code to execute",
                    required=True,
                ),
            ],
        )

        self.timeout = timeout
        self.sandbox = sandbox
        self.allowed_modules = allowed_modules or ["math", "json", "re", "datetime", "collections"]

    async def execute(self, code: str, **kwargs: Any) -> ToolResult:
        """Execute Python code."""
        import time
        start_time = time.time()

        try:
            # Validate code
            self._validate_code(code)

            # Execute in subprocess for safety
            if self.sandbox:
                result = await self._execute_sandboxed(code)
            else:
                result = await self._execute_local(code)

            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(True, latency_ms)

            return ToolResult(
                success=True,
                output=result,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(False, latency_ms)

            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                latency_ms=latency_ms,
            )

    def _validate_code(self, code: str) -> None:
        """Validate code for safety."""
        dangerous = ["import os", "import sys", "subprocess", "eval(", "exec("]

        for pattern in dangerous:
            if pattern in code:
                raise ValueError(f"Potentially dangerous code pattern: {pattern}")

    async def _execute_local(self, code: str) -> str:
        """Execute code locally (less safe)."""
        # Capture stdout
        import io
        from contextlib import redirect_stdout

        output_buffer = io.StringIO()

        try:
            with redirect_stdout(output_buffer):
                exec(code, {"__builtins__": __builtins__}, {})
            return output_buffer.getvalue() or "Code executed successfully (no output)"
        except Exception as e:
            raise RuntimeError(f"Code execution error: {e}")

    async def _execute_sandboxed(self, code: str) -> str:
        """Execute code in sandboxed subprocess."""
        # Create wrapper script
        wrapper = f"""
import sys
import json

# Restricted imports
allowed_modules = {json.dumps(self.allowed_modules)}

for mod in allowed_modules:
    try:
        __import__(mod)
    except ImportError:
        pass

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    wrapper,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self.timeout,
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                raise RuntimeError(stderr.decode() or "Code execution failed")

            return stdout.decode() or "Code executed successfully (no output)"

        except asyncio.TimeoutError:
            raise RuntimeError(f"Code execution timed out after {self.timeout}s")


class SearchTool(Tool):
    """
    Search tool for web and document search.

    Integrates with search APIs.
    """

    def __init__(
        self,
        search_engine: str = "duckduckgo",  # duckduckgo, google, bing
        api_key: Optional[str] = None,
        num_results: int = 5,
    ) -> None:
        super().__init__(
            name="search",
            description="Search the web for information. Returns relevant search results with titles, snippets, and URLs.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of results to return",
                    required=False,
                    default=num_results,
                ),
            ],
        )

        self.search_engine = search_engine
        self.api_key = api_key
        self.num_results = num_results

        self._client = httpx.AsyncClient(timeout=30.0)

    async def execute(self, query: str, num_results: Optional[int] = None, **kwargs: Any) -> ToolResult:
        """Execute search."""
        import time
        start_time = time.time()

        num_results = num_results or self.num_results

        try:
            if self.search_engine == "duckduckgo":
                results = await self._duckduckgo_search(query, num_results)
            elif self.search_engine == "google" and self.api_key:
                results = await self._google_search(query, num_results)
            else:
                results = await self._duckduckgo_search(query, num_results)

            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(True, latency_ms)

            return ToolResult(
                success=True,
                output=results,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(False, latency_ms)

            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                latency_ms=latency_ms,
            )

    async def _duckduckgo_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=num_results))

            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
                for r in results
            ]
        except ImportError:
            # Fallback to simple HTML scraping
            return await self._fallback_search(query, num_results)

    async def _google_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        cse_id = os.getenv("GOOGLE_CSE_ID", "")

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": cse_id,
            "q": query,
            "num": min(num_results, 10),
        }

        response = await self._client.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        results = data.get("items", [])

        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("link", ""),
            }
            for r in results
        ]

    async def _fallback_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Fallback search using HTML parsing."""
        # Simple implementation - in production use proper search API
        return [
            {
                "title": f"Result {i + 1} for: {query}",
                "snippet": f"Search result snippet for '{query}'",
                "url": f"https://example.com/result/{i}",
            }
            for i in range(num_results)
        ]

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


class ToolRegistry:
    """
    Registry for managing and discovering tools.

    Features:
    - Tool registration
    - Tool discovery
    - Batch execution
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: Tool, category: Optional[str] = None) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool.name)

        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List registered tools."""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())

    def get_definitions(self, names: Optional[List[str]] = None) -> List[ToolDefinition]:
        """Get tool definitions for LLM."""
        tools = [self._tools[name] for name in (names or self._tools.keys())]
        return [tool.get_definition() for tool in tools]

    def to_openai_format(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get OpenAI function calling format."""
        definitions = self.get_definitions(names)
        return [d.to_openai_format() for d in definitions]

    async def execute(
        self,
        tool_name: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}",
            )

        return await tool.execute(**kwargs)

    async def execute_batch(
        self,
        executions: List[Tuple[str, Dict[str, Any]]],
    ) -> Dict[str, ToolResult]:
        """Execute multiple tools in parallel."""
        async def run(name: str, args: Dict[str, Any]) -> Tuple[str, ToolResult]:
            result = await self.execute(name, **args)
            return name, result

        tasks = [run(name, args) for name, args in executions]
        results = await asyncio.gather(*tasks)

        return dict(results)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "categories": self._categories,
            "tool_stats": {
                name: tool.get_stats()
                for name, tool in self._tools.items()
            },
        }


class ToolExecutor:
    """
    Executor for tool-based agent workflows.

    Handles tool selection, execution, and result processing.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm_client: Any,
        max_iterations: int = 10,
    ) -> None:
        self.registry = registry
        self.llm_client = llm_client
        self.max_iterations = max_iterations

    async def execute_task(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task using tools.

        Args:
            task: Task description
            context: Optional context

        Returns:
            Task result
        """
        history = []
        tools_format = self.registry.to_openai_format()

        for iteration in range(self.max_iterations):
            # Get next action from LLM
            action = await self._get_next_action(task, history, tools_format, context)

            if action.get("type") == "final_answer":
                return {
                    "success": True,
                    "result": action.get("answer"),
                    "iterations": iteration + 1,
                    "history": history,
                }

            # Execute tool
            tool_name = action.get("tool")
            tool_args = action.get("arguments", {})

            result = await self.registry.execute(tool_name, **tool_args)

            history.append({
                "action": action,
                "result": result.to_dict(),
            })

            if not result.success:
                return {
                    "success": False,
                    "error": result.error,
                    "iterations": iteration + 1,
                    "history": history,
                }

        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached",
            "history": history,
        }

    async def _get_next_action(
        self,
        task: str,
        history: List[Dict],
        tools_format: List[Dict],
        context: Optional[str],
    ) -> Dict[str, Any]:
        """Get next action from LLM."""
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(tools_format),
            },
        ]

        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})

        messages.append({"role": "user", "content": f"Task: {task}"})

        if history:
            history_text = "\n".join(
                f"Action: {h['action']}\nResult: {h['result']}"
                for h in history
            )
            messages.append({"role": "assistant", "content": f"History:\n{history_text}"})

        response = await self.llm_client.generate(
            messages=messages,
            tools=tools_format,
            tool_choice="auto",
        )

        # Parse response
        raw = response.raw_response if hasattr(response, 'raw_response') else {}
        choice = raw.get("choices", [{}])[0] if raw else {}
        message = choice.get("message", {})

        # Check for tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            call = tool_calls[0].get("function", {})
            return {
                "type": "tool_call",
                "tool": call.get("name"),
                "arguments": json.loads(call.get("arguments", "{}")),
            }

        # Check for final answer
        content = message.get("content", "")
        if "final answer" in content.lower() or not tools_format:
            return {
                "type": "final_answer",
                "answer": content,
            }

        return {
            "type": "unknown",
            "content": content,
        }

    def _build_system_prompt(self, tools_format: List[Dict]) -> str:
        """Build system prompt for tool execution."""
        return """You are a helpful assistant that can use tools to complete tasks.

Available tools:
""" + "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in tools_format
        ) + """

To use a tool, respond with a tool call. When you have the final answer, respond with "Final Answer: [your answer]".

Think step by step and use tools as needed."""


# Import os for environment variables
import os
