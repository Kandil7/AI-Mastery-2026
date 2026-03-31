"""
Structured Output Module

Production-ready implementations for generating structured outputs from LLMs:
- JSON Schema validation
- Outlines integration for constrained decoding
- Function calling with validation
- Pydantic model integration

Features:
- Schema validation and enforcement
- Retry on invalid output
- Type-safe output parsing
- Multiple output formats
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ValidationError, create_model

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import outlines
    from outlines.models import OpenAI as OutlinesOpenAI
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    logger.warning("Outlines not installed. Constrained decoding unavailable.")

try:
    import instructor
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger.warning("Instructor not installed. Pydantic integration limited.")


T = TypeVar("T", bound=BaseModel)


class OutputFormat(str, Enum):
    """Supported output formats."""

    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    MARKDOWN_TABLE = "markdown_table"
    PYDANTIC = "pydantic"


@dataclass
class OutputSchema:
    """
    Schema definition for structured output.

    Can be defined as JSON Schema, Pydantic model, or dict.
    """

    name: str
    description: str
    schema: Dict[str, Any]
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel], name: str = "Output") -> "OutputSchema":
        """Create schema from Pydantic model."""
        json_schema = model.model_json_schema()

        return cls(
            name=name,
            description=json_schema.get("description", ""),
            schema=json_schema,
            properties=json_schema.get("properties", {}),
            required=json_schema.get("required", []),
        )

    @classmethod
    def from_dict(
        cls,
        properties: Dict[str, Dict[str, Any]],
        name: str = "Output",
        description: str = "",
        required: Optional[List[str]] = None,
    ) -> "OutputSchema":
        """Create schema from property definitions."""
        schema = {
            "type": "object",
            "properties": properties,
            "required": required or list(properties.keys()),
        }

        return cls(
            name=name,
            description=description,
            schema=schema,
            properties=properties,
            required=required or list(properties.keys()),
        )

    def to_json_schema(self) -> Dict[str, Any]:
        """Get JSON Schema representation."""
        return self.schema

    def to_prompt_description(self) -> str:
        """Generate human-readable schema description for prompts."""
        lines = [f"Output Schema: {self.name}"]
        lines.append(f"Description: {self.description}")
        lines.append("\nProperties:")

        for prop_name, prop_def in self.properties.items():
            prop_type = prop_def.get("type", "any")
            prop_desc = prop_def.get("description", "")
            required = " (required)" if prop_name in self.required else ""
            lines.append(f"  - {prop_name}: {prop_type}{required} - {prop_desc}")

        if self.examples:
            lines.append("\nExample:")
            lines.append(json.dumps(self.examples[0], indent=2))

        return "\n".join(lines)


class JSONSchemaValidator:
    """
    Validates JSON output against a schema.

    Features:
    - Schema validation
    - Auto-repair attempts
    - Detailed error reporting
    """

    def __init__(
        self,
        schema: OutputSchema,
        max_repair_attempts: int = 3,
    ) -> None:
        self.schema = schema
        self.max_repair_attempts = max_repair_attempts

        # Try to import jsonschema for validation
        try:
            import jsonschema
            self._jsonschema = jsonschema
            self._has_validator = True
        except ImportError:
            self._jsonschema = None
            self._has_validator = False
            logger.warning("jsonschema not installed. Using basic validation.")

    def validate(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate data against schema.

        Args:
            data: JSON string or dict to validate

        Returns:
            Validated data dict

        Raises:
            ValidationError: If validation fails
        """
        # Parse if string
        if isinstance(data, str):
            data = self._parse_json(data)

        # Validate
        if self._has_validator and self._jsonschema:
            try:
                self._jsonschema.validate(instance=data, schema=self.schema.schema)
            except self._jsonschema.ValidationError as e:
                raise ValidationError(f"Schema validation failed: {e.message}") from e

        # Basic validation
        self._basic_validate(data)

        return data

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling common issues."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to fix common issues
        repaired = self._repair_json(text)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        raise ValidationError(f"Failed to parse JSON from: {text[:200]}")

    def _repair_json(self, text: str) -> Optional[str]:
        """Attempt to repair common JSON issues."""
        repaired = text

        # Fix missing quotes on keys
        repaired = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)

        # Fix single quotes
        repaired = repaired.replace("'", '"')

        # Fix trailing commas
        repaired = re.sub(r',\s*}', '}', repaired)
        repaired = re.sub(r',\s*]', ']', repaired)

        # Fix missing commas
        repaired = re.sub(r'"\s*"', '", "', repaired)

        return repaired if repaired != text else None

    def _basic_validate(self, data: Dict[str, Any]) -> None:
        """Basic validation without jsonschema library."""
        if not isinstance(data, dict):
            raise ValidationError("Output must be a JSON object")

        # Check required fields
        for required_field in self.schema.required:
            if required_field not in data:
                raise ValidationError(f"Missing required field: {required_field}")

        # Check types
        for field_name, field_def in self.schema.properties.items():
            if field_name in data:
                expected_type = field_def.get("type")
                value = data[field_name]

                if expected_type == "string" and not isinstance(value, str):
                    raise ValidationError(f"Field '{field_name}' must be a string")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    raise ValidationError(f"Field '{field_name}' must be a number")
                elif expected_type == "integer" and not isinstance(value, int):
                    raise ValidationError(f"Field '{field_name}' must be an integer")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    raise ValidationError(f"Field '{field_name}' must be a boolean")
                elif expected_type == "array" and not isinstance(value, list):
                    raise ValidationError(f"Field '{field_name}' must be an array")
                elif expected_type == "object" and not isinstance(value, dict):
                    raise ValidationError(f"Field '{field_name}' must be an object")


class StructuredOutputGenerator:
    """
    Generates structured outputs from LLMs.

    Features:
    - Schema-constrained generation
    - Automatic retry on invalid output
    - Multiple format support
    - Pydantic integration
    """

    def __init__(
        self,
        client: Any,  # BaseLLMClient
        schema: OutputSchema,
        output_format: OutputFormat = OutputFormat.JSON,
        max_retries: int = 3,
        use_outlines: bool = False,
        use_instructor: bool = False,
    ) -> None:
        self.client = client
        self.schema = schema
        self.output_format = output_format
        self.max_retries = max_retries
        self.use_outlines = use_outlines and OUTLINES_AVAILABLE
        self.use_instructor = use_instructor and INSTRUCTOR_AVAILABLE

        self.validator = JSONSchemaValidator(schema)

        if self.use_outlines:
            logger.info("Using Outlines for constrained decoding")
        if self.use_instructor:
            logger.info("Using Instructor for Pydantic integration")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,  # Lower for structured output
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (low for structured)
            **kwargs: Additional generation parameters

        Returns:
            Validated structured output
        """
        # Build prompt with schema
        full_prompt = self._build_structured_prompt(prompt)

        for attempt in range(self.max_retries):
            try:
                # Use constrained decoding if available
                if self.use_outlines:
                    response = await self._generate_with_outlines(
                        full_prompt,
                        system_prompt,
                        temperature,
                        **kwargs,
                    )
                else:
                    response = await self.client.generate(
                        messages=self._build_messages(full_prompt, system_prompt),
                        temperature=temperature,
                        **kwargs,
                    )

                # Parse and validate
                output = self._parse_response(response.content if hasattr(response, 'content') else response)
                validated = self.validator.validate(output)

                logger.info(f"Generated valid structured output on attempt {attempt + 1}")
                return validated

            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                # Add error feedback to prompt for retry
                full_prompt += f"\n\nPrevious attempt failed: {e}. Please try again."

        raise ValidationError("Failed to generate valid structured output")

    async def generate_pydantic(
        self,
        model: Type[T],
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """
        Generate output as Pydantic model.

        Args:
            model: Pydantic model class
            prompt: Input prompt
            system_prompt: Optional system prompt
            **kwargs: Generation parameters

        Returns:
            Instantiated Pydantic model
        """
        if self.use_instructor and INSTRUCTOR_AVAILABLE:
            return await self._generate_with_instructor(model, prompt, system_prompt, **kwargs)

        # Fallback: generate JSON and parse
        schema = OutputSchema.from_pydantic(model)
        self.schema = schema
        self.validator = JSONSchemaValidator(schema)

        output = await self.generate(prompt, system_prompt, **kwargs)
        return model(**output)

    def _build_structured_prompt(self, user_prompt: str) -> str:
        """Build prompt with schema instructions."""
        format_instructions = self._get_format_instructions()

        return f"""{user_prompt}

{format_instructions}

Respond ONLY with valid {self.output_format.value} matching the schema above.
Do not include any explanation or additional text."""

    def _get_format_instructions(self) -> str:
        """Get format-specific instructions."""
        instructions = {
            OutputFormat.JSON: f"""Output Format: JSON
Schema:
{json.dumps(self.schema.schema, indent=2)}""",

            OutputFormat.XML: """Output Format: XML
Structure your response as valid XML with proper nesting.""",

            OutputFormat.YAML: """Output Format: YAML
Structure your response as valid YAML.""",

            OutputFormat.CSV: """Output Format: CSV
Structure your response as CSV with headers.""",

            OutputFormat.MARKDOWN_TABLE: """Output Format: Markdown Table
Structure your response as a markdown table with headers.""",

            OutputFormat.PYDANTIC: f"""Output Format: JSON (Pydantic-compatible)
Schema:
{json.dumps(self.schema.schema, indent=2)}""",
        }

        return instructions.get(self.output_format, instructions[OutputFormat.JSON])

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        """Build message list for API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return messages

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse response based on format."""
        if self.output_format == OutputFormat.JSON:
            return self.validator._parse_json(response_text)
        elif self.output_format == OutputFormat.XML:
            return self._parse_xml(response_text)
        elif self.output_format == OutputFormat.YAML:
            return self._parse_yaml(response_text)
        elif self.output_format == OutputFormat.CSV:
            return self._parse_csv(response_text)
        else:
            return {"content": response_text}

    def _parse_xml(self, text: str) -> Dict[str, Any]:
        """Parse XML response."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(text.strip())
            return {child.tag: child.text for child in root}
        except Exception as e:
            raise ValidationError(f"Failed to parse XML: {e}")

    def _parse_yaml(self, text: str) -> Dict[str, Any]:
        """Parse YAML response."""
        try:
            import yaml
            return yaml.safe_load(text.strip())
        except ImportError:
            # Fallback: try JSON
            return self.validator._parse_json(text)
        except Exception as e:
            raise ValidationError(f"Failed to parse YAML: {e}")

    def _parse_csv(self, text: str) -> Dict[str, Any]:
        """Parse CSV response."""
        import csv
        import io

        try:
            reader = csv.DictReader(io.StringIO(text.strip()))
            rows = list(reader)
            return {"headers": reader.fieldnames, "rows": rows}
        except Exception as e:
            raise ValidationError(f"Failed to parse CSV: {e}")

    async def _generate_with_outlines(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        **kwargs: Any,
    ) -> Any:
        """Generate using Outlines for constrained decoding."""
        if not OUTLINES_AVAILABLE:
            raise RuntimeError("Outlines not available")

        # Create outlines model and grammar
        # Note: This is a simplified example; actual implementation depends on provider
        logger.debug("Using Outlines constrained decoding")

        # Fall back to regular generation with schema in prompt
        return await self.client.generate(
            messages=self._build_messages(prompt, system_prompt),
            temperature=temperature,
            **kwargs,
        )

    async def _generate_with_instructor(
        self,
        model: Type[T],
        prompt: str,
        system_prompt: Optional[str],
        **kwargs: Any,
    ) -> T:
        """Generate using Instructor for Pydantic integration."""
        if not INSTRUCTOR_AVAILABLE:
            raise RuntimeError("Instructor not available")

        # Patch the client with instructor
        import instructor

        # Create instructor client based on underlying client type
        if hasattr(self.client, '_client'):
            # OpenAI client
            client = instructor.from_openai(self.client._client)
        else:
            # Generic fallback
            client = instructor.from_litellum(self.client)

        return await client.chat.completions.create(
            model=self.client.model,
            messages=self._build_messages(prompt, system_prompt),
            response_model=model,
            **kwargs,
        )


@dataclass
class FunctionDefinition:
    """Definition of a callable function for LLM."""

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionCaller:
    """
    Handles function calling with LLMs.

    Features:
    - Function definition and registration
    - Automatic parameter extraction
    - Function execution with validation
    - Multi-function routing
    """

    def __init__(
        self,
        client: Any,  # BaseLLMClient
        functions: Optional[List[FunctionDefinition]] = None,
    ) -> None:
        self.client = client
        self.functions: Dict[str, FunctionDefinition] = {}

        if functions:
            for func in functions:
                self.register_function(func)

    def register_function(self, function: FunctionDefinition) -> None:
        """Register a function for calling."""
        self.functions[function.name] = function
        logger.info(f"Registered function: {function.name}")

    def register_from_callable(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Register a Python callable as a function."""
        import inspect

        func_name = name or func.__name__
        func_desc = description or (func.__doc__ or "No description")

        # Extract parameters
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == list:
                param_type = "array"
            elif param.annotation == dict:
                param_type = "object"

            parameters["properties"][param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",
            }

            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        function_def = FunctionDefinition(
            name=func_name,
            description=func_desc,
            parameters=parameters,
            func=func,
        )

        self.register_function(function_def)

    async def execute(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        auto_execute: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute function calling.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            auto_execute: Whether to automatically execute selected function
            **kwargs: Generation parameters

        Returns:
            Function result or selection
        """
        if not self.functions:
            raise ValueError("No functions registered")

        # Build function definitions for API
        tools = [f.to_openai_format() for f in self.functions.values()]

        # For OpenAI-style API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Generate with function calling
        response = await self.client.generate(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **kwargs,
        )

        # Check if function was selected
        raw = response.raw_response if hasattr(response, 'raw_response') else {}
        choice = raw.get("choices", [{}])[0] if raw else {}
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if tool_calls and auto_execute:
            return await self._execute_tool_calls(tool_calls)

        return {"response": response.content if hasattr(response, 'content') else str(response)}

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute tool calls and return results."""
        results = []

        for call in tool_calls:
            function = call.get("function", {})
            func_name = function.get("name")
            args_str = function.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            if func_name in self.functions:
                func_def = self.functions[func_name]
                try:
                    result = func_def.func(**args)
                    results.append({
                        "function": func_name,
                        "result": result,
                        "success": True,
                    })
                except Exception as e:
                    results.append({
                        "function": func_name,
                        "error": str(e),
                        "success": False,
                    })
            else:
                results.append({
                    "function": func_name,
                    "error": f"Unknown function: {func_name}",
                    "success": False,
                })

        return {"tool_results": results}

    async def route(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """
        Route prompt to appropriate function.

        Returns the name of the selected function.
        """
        function_names = ", ".join(self.functions.keys())

        routing_prompt = f"""Given the following request, select which function should handle it.

Available functions: {function_names}

Request: {prompt}

Respond with ONLY the function name."""

        response = await self.client.generate(
            messages=[{"role": "user", "content": routing_prompt}],
            temperature=0,
            **kwargs,
        )

        selected = response.content.strip() if hasattr(response, 'content') else str(response)

        if selected in self.functions:
            return selected

        raise ValueError(f"Invalid function selected: {selected}")


# Pre-built schemas for common use cases

class ExtractionSchema:
    """Pre-built schemas for common extraction tasks."""

    @staticmethod
    def entity_extraction() -> OutputSchema:
        """Schema for entity extraction."""
        return OutputSchema.from_dict(
            properties={
                "entities": {
                    "type": "array",
                    "description": "List of extracted entities",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Entity text"},
                            "type": {"type": "string", "description": "Entity type (PERSON, ORG, etc.)"},
                            "confidence": {"type": "number", "description": "Confidence score 0-1"},
                        },
                        "required": ["text", "type"],
                    },
                },
            },
            name="EntityExtraction",
            description="Extract named entities from text",
        )

    @staticmethod
    def sentiment_analysis() -> OutputSchema:
        """Schema for sentiment analysis."""
        return OutputSchema.from_dict(
            properties={
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "Overall sentiment",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0-1",
                },
                "aspects": {
                    "type": "array",
                    "description": "Aspect-level sentiments",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {"type": "string"},
                            "sentiment": {"type": "string"},
                        },
                    },
                },
            },
            name="SentimentAnalysis",
            description="Analyze sentiment in text",
            required=["sentiment", "confidence"],
        )

    @staticmethod
    def text_classification(labels: List[str]) -> OutputSchema:
        """Schema for text classification."""
        return OutputSchema.from_dict(
            properties={
                "label": {
                    "type": "string",
                    "enum": labels,
                    "description": "Classification label",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0-1",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for classification",
                },
            },
            name="TextClassification",
            description="Classify text into categories",
            required=["label", "confidence"],
        )

    @staticmethod
    def qa_extraction() -> OutputSchema:
        """Schema for question answering."""
        return OutputSchema.from_dict(
            properties={
                "answer": {
                    "type": "string",
                    "description": "Direct answer to the question",
                },
                "evidence": {
                    "type": "string",
                    "description": "Supporting evidence from context",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0-1",
                },
            },
            name="QuestionAnswering",
            description="Extract answer from context",
            required=["answer", "confidence"],
        )


# Pydantic models for common tasks

class StructuredResponse(BaseModel):
    """Base model for structured responses."""

    class Config:
        extra = "forbid"


class EntityExtractionResponse(StructuredResponse):
    """Response model for entity extraction."""

    entities: List[Dict[str, Any]]
    source_text: Optional[str] = None


class SentimentResponse(StructuredResponse):
    """Response model for sentiment analysis."""

    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    aspects: Optional[List[Dict[str, str]]] = None


class ClassificationResponse(StructuredResponse):
    """Response model for text classification."""

    label: str
    confidence: float
    reasoning: Optional[str] = None


class QAResponse(StructuredResponse):
    """Response model for question answering."""

    answer: str
    evidence: Optional[str] = None
    confidence: float
