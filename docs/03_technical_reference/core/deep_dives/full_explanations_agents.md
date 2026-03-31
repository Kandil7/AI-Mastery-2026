# Agents package (agents/*.py)

## agents/executor.py

### AgentResult (dataclass)
- Purpose: container for the agent run output and aggregated citations.
- Fields:
  - `output: str` -> concatenated tool outputs.
  - `citations: List[dict]` -> citations collected from tools.

### AgentExecutor
- Purpose: executes a multi-step plan using the registered tools.
- `__init__(planner: Planner, tools: ToolRegistry)`
  - Stores the planner and tool registry for later use.
- `run(question: str) -> AgentResult`
  - Calls `planner.plan(question)` to produce tool steps.
  - Executes each step via `ToolRegistry.run(...)`.
  - Aggregates text outputs and citations.
  - Returns `AgentResult` with joined output text and citations.

## agents/planner.py

### PlanStep (dataclass)
- Purpose: represents a single tool invocation.
- Fields:
  - `tool: str` -> tool name (e.g., `rag`, `sql`, `web`).
  - `input: str` -> input text for that tool.

### Planner
- Purpose: simple heuristic planner that decides which tools to use.
- `plan(question: str) -> List[PlanStep]`
  - Inspects the question string.
  - If it contains `sql:` or `database`, includes a SQL step.
  - If it contains `web:` or `external`, includes a Web step.
  - Always includes a RAG step at the end.

## agents/policies.py

### RoutingPolicy (dataclass)
- Purpose: picks model provider based on task type.
- Fields:
  - `default_provider: str` -> primary provider name.
  - `fallback_provider: str` -> backup provider name.
- `choose(task: str) -> str`
  - Uses simple keyword checks to decide provider.
  - Returns fallback if task is not summary/extraction.

## agents/tooling.py

### build_rag_tool() -> Tool
- Purpose: constructs a RAG tool used by the agent.
- Behavior:
  - Loads settings.
  - Creates vector store and embeddings provider.
  - Retrieves chunks with `HybridRetriever` (vector-only here).
  - Reranks with `Reranker`.
  - Generates answer with `generate_answer`.
  - Formats citations with `format_citations`.

### build_sql_tool(dsn: str, query_template: str | None) -> Tool
- Purpose: constructs a SQL tool for structured data retrieval.
- Behavior:
  - Creates a SQLAlchemy engine.
  - Executes a template or raw question as SQL.
  - Returns rows as a string.

### build_web_tool(base_url: str, headers: Dict[str, Any] | None) -> Tool
- Purpose: constructs a Web tool for external search/lookup.
- Behavior:
  - Calls `GET base_url?q=question` with optional headers.
  - Returns raw response text.

## agents/tools.py

### Tool (dataclass)
- Purpose: describes a callable tool.
- Fields:
  - `name: str` -> tool identifier.
  - `handler: Callable[[str], ToolResult]` -> callable that executes the tool.

### ToolResult (dataclass)
- Purpose: standardized output of a tool.
- Fields:
  - `output: str` -> tool output text.
  - `citations: List[Dict[str, Any]]` -> citations used by the tool.
  - `metadata: Dict[str, Any]` -> optional metadata.

### ToolRegistry
- Purpose: registry and dispatcher for tools.
- `register(tool: Tool) -> None`
  - Registers a tool by name.
- `run(name: str, input_text: str) -> ToolResult`
  - Finds the tool and executes its handler.

## agents/verifier.py

### Verifier
- Purpose: checks whether an answer is supported by citations.
- `__init__(provider: LLMProvider | None)`
  - Stores provider; if None, verification is bypassed.
- `verify(answer: str, citations: List[dict]) -> bool`
  - Prompts the model to answer YES/NO.
  - Returns True if response starts with "yes".
