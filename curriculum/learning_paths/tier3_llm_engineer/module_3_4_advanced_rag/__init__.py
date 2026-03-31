"""
Module 3.4: Advanced RAG

Production-ready advanced RAG implementations:
- Query Construction: SQL, Cypher, metadata filters, query translation
- Tools & Agents: Tool integration, API calls, code interpreter
- Post-Processing: Re-ranking, RAG-fusion, diversity, synthesis
- Program LLM: DSPy integration, prompt optimization
"""

from .query_construction import (
    QueryConstructor,
    SQLConstructor,
    CypherConstructor,
    MetadataFilterConstructor,
    QueryTranslator,
)
from .tools_agents import (
    ToolRegistry,
    Tool,
    APITool,
    CodeInterpreter,
    CalculatorTool,
    SearchTool,
    ToolExecutor,
)
from .post_processing import (
    PostProcessor,
    Reranker,
    RAGFusion,
    DiversityEnhancer,
    AnswerSynthesizer,
)
from .program_llm import (
    ProgramLLM,
    DSPyWrapper,
    PromptOptimizer,
    Bootstrapper,
    CompiledProgram,
)

__all__ = [
    # Query Construction
    "QueryConstructor",
    "SQLConstructor",
    "CypherConstructor",
    "MetadataFilterConstructor",
    "QueryTranslator",
    # Tools & Agents
    "ToolRegistry",
    "Tool",
    "APITool",
    "CodeInterpreter",
    "CalculatorTool",
    "SearchTool",
    "ToolExecutor",
    # Post-Processing
    "PostProcessor",
    "Reranker",
    "RAGFusion",
    "DiversityEnhancer",
    "AnswerSynthesizer",
    # Program LLM
    "ProgramLLM",
    "DSPyWrapper",
    "PromptOptimizer",
    "Bootstrapper",
    "CompiledProgram",
]

__version__ = "1.0.0"
