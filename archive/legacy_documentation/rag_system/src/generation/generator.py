"""
Generation Layer - LLM Integration for Arabic Islamic Literature RAG

Following RAG Pipeline Guide 2026 - Phase 6: Generation & Response
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    MOCK = "mock"  # For testing


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    answer: str
    citations: List[Dict[str, Any]]
    tokens_used: int
    latency_ms: float
    model: str


@dataclass
class RAGPrompt:
    """Structured prompt for RAG generation."""

    system_prompt: str
    user_prompt: str
    context: str


class ArabicPrompts:
    """
    Prompt templates optimized for Arabic Islamic literature.
    """

    # System prompt for Arabic Islamic literature assistant
    SYSTEM_PROMPT_ARABIC = """أنت باحث ومتخصص في التراث الإسلامي والأدب العربي. 
لديك الوصول إلى مجموعة ضخمة من الكتب الإسلامية الكلاسيكية، 包括 التفسير والحديث والفقه والعقيدة والأدب.

عند الإجابة على الأسئلة:
1. استخدم فقط المعلومات المقدمة من المصادر أدناه
2. استشهد بالمصادر المحددة (اسم الكتاب، المؤلف) عند الإمكان
3. إذا لم تحتوي المعلومات المتاحة على إجابة كافية، اعترف بذلك بوضوح
4. حافظ على الدقة العلمية والاستشهاد بالمراجع
5. أجب بالعربية عندما تكون السؤال بالعربية، وبالإنجليزية otherwise

ملاحظة مهمة: لا تخترع معلومات. إذا لم تجد إجابة في السياق المقدم، قل ذلك بوضوح."""

    # English system prompt (alternative)
    SYSTEM_PROMPT_ENGLISH = """You are a knowledgeable assistant specializing in Arabic Islamic literature.
You have access to a vast collection of Islamic books, hadith commentaries,
tafsir (Quranic exegesis), fiqh (jurisprudence), theology, and classical Arabic literature.

When answering questions:
1. Base your answers ONLY on the provided context from the source texts
2. Cite the specific sources (book title, author) when possible
3. If the context doesn't contain enough information, acknowledge the limitation clearly
4. Maintain scholarly accuracy and cite references properly
5. If you're not sure about something, say so honestly

Important: Do not make up information. If you cannot find the answer in the provided context, say so clearly."""

    @staticmethod
    def build_context(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
        """Build context from retrieved chunks."""

        context_parts = []

        for i, chunk in enumerate(chunks[:max_chunks], 1):
            metadata = chunk.get("metadata", {})

            source_info = f"[المصدر {i}]"
            if metadata.get("book_title"):
                source_info += f" - {metadata['book_title']}"
            if metadata.get("author"):
                source_info += f" - تأليف: {metadata['author']}"
            if metadata.get("category"):
                source_info += f" ({metadata['category']})"

            content = chunk.get("content", "")

            context_parts.append(f"""
{source_info}
{content}
---""")

        return "\n\n".join(context_parts)

    @staticmethod
    def build_user_prompt(query: str, context: str, language: str = "auto") -> str:
        """Build user prompt with context."""

        if language == "arabic" or (
            language == "auto" and ArabicPrompts._is_arabic(query)
        ):
            instructions = """تعليمات:
1. أجب باستخدام المعلومات المقدمة فقط من السياق أعلاه
2. استشهد بالمصادر باستخدام الأرقام [1], [2], etc.
3. كن موجزاً لكن كاملاً
4. إذا لم تجد الإجابة في السياق، قل "لا تتوفر معلومات كافية في السياق للإجابة على هذا السؤال" """
        else:
            instructions = """Instructions:
1. Answer using ONLY the context provided above
2. Cite your sources using [1], [2], etc.
3. Be concise but complete
4. If the answer is not in the context, say "I don't have sufficient information in the provided context to answer this question"
5. If you're unsure, acknowledge the uncertainty"""

        return f"""## Context

{context}

## {instructions}

## Question

{query}

## Answer

"""

    @staticmethod
    def _is_arabic(text: str) -> bool:
        """Check if text contains Arabic characters."""
        import re

        return bool(re.search(r"[\u0600-\u06FF]", text))


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

        # Initialize client
        self._client = self._init_client()

    def _init_client(self):
        """Initialize the LLM client."""

        if self.provider == LLMProvider.OPENAI:
            try:
                from openai import AsyncOpenAI

                return AsyncOpenAI()
            except ImportError:
                logger.warning("openai not installed, using mock client")
                return None

        elif self.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic

                return anthropic.AsyncAnthropic()
            except ImportError:
                logger.warning("anthropic not installed, using mock client")
                return None

        elif self.provider == LLMProvider.OLLAMA:
            return self._OllamaClient()

        else:
            return None  # Mock client

    async def generate(
        self,
        prompt: str,
        stream: bool = False,
    ) -> GenerationResult:
        """Generate response from LLM."""

        start_time = time.time()

        if self.provider == LLMProvider.OPENAI:
            return await self._generate_openai(prompt, stream, start_time)
        elif self.provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic(prompt, stream, start_time)
        elif self.provider == LLMProvider.OLLAMA:
            return await self._generate_ollama(prompt, stream, start_time)
        else:
            return await self._generate_mock(prompt, start_time)

    async def _generate_openai(
        self,
        prompt: str,
        stream: bool,
        start_time: float,
    ) -> GenerationResult:
        """Generate using OpenAI API."""

        if self._client is None:
            return await self._generate_mock(prompt, start_time)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
            )

            if stream:
                # Handle streaming (return first chunk as demo)
                content = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
            else:
                content = response.choices[0].message.content

            latency_ms = (time.time() - start_time) * 1000
            tokens_used = (
                response.usage.total_tokens if hasattr(response, "usage") else 0
            )

            return GenerationResult(
                answer=content,
                citations=[],
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                model=self.model,
            )

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return await self._generate_mock(prompt, start_time)

    async def _generate_anthropic(
        self,
        prompt: str,
        stream: bool,
        start_time: float,
    ) -> GenerationResult:
        """Generate using Anthropic API."""

        if self._client is None:
            return await self._generate_mock(prompt, start_time)

        try:
            response = await self._client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
            )

            content = response.content[0].text

            latency_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                answer=content,
                citations=[],
                tokens_used=0,  # Anthropic doesn't provide this in same way
                latency_ms=latency_ms,
                model=self.model,
            )

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return await self._generate_mock(prompt, start_time)

    async def _generate_ollama(
        self,
        prompt: str,
        stream: bool,
        start_time: float,
    ) -> GenerationResult:
        """Generate using Ollama (local)."""

        try:
            response = await self._client.generate(
                model=self.model,
                prompt=prompt,
                stream=stream,
            )

            content = response.get("response", "")

            latency_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                answer=content,
                citations=[],
                tokens_used=0,
                latency_ms=latency_ms,
                model=self.model,
            )

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return await self._generate_mock(prompt, start_time)

    async def _generate_mock(
        self,
        prompt: str,
        start_time: float,
    ) -> GenerationResult:
        """Mock generation for testing."""

        # Simulate latency
        await asyncio.sleep(0.1)

        latency_ms = (time.time() - start_time) * 1000

        # Extract question from prompt
        question = "the question"
        if "## Question" in prompt:
            question_start = prompt.find("## Question") + len("## Question")
            question_section = prompt[question_start:]
            if "## Answer" in question_section:
                question = question_section[
                    : question_section.find("## Answer")
                ].strip()

        return GenerationResult(
            answer=f"[This is a mock response for: {question}]\n\nPlease configure an LLM provider (OpenAI, Anthropic, or Ollama) to get actual responses.",
            citations=[],
            tokens_used=0,
            latency_ms=latency_ms,
            model="mock",
        )

    async def generate_stream(
        self,
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""

        if self.provider == LLMProvider.OPENAI and self._client:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        else:
            # Fallback to non-streaming
            result = await self.generate(prompt)
            yield result.answer


class RAGGenerator:
    """
    Complete RAG generation pipeline.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        include_citations: bool = True,
        max_context_chunks: int = 5,
        language_detection: bool = True,
    ):
        self.llm = llm_client
        self.include_citations = include_citations
        self.max_context_chunks = max_context_chunks
        self.language_detection = language_detection
        self.prompts = ArabicPrompts()

    async def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        stream: bool = False,
    ) -> GenerationResult:
        """
        Generate answer from query and retrieved context.

        Args:
            query: User question
            retrieved_chunks: Retrieved context chunks
            stream: Whether to stream the response

        Returns:
            GenerationResult with answer and metadata
        """

        # Detect language
        language = "auto"
        if self.language_detection:
            if self.prompts._is_arabic(query):
                language = "arabic"
            else:
                language = "english"

        # Build context
        context = self.prompts.build_context(
            retrieved_chunks, max_chunks=self.max_context_chunks
        )

        # Build prompts
        if language == "arabic":
            system_prompt = self.prompts.SYSTEM_PROMPT_ARABIC
        else:
            system_prompt = self.prompts.SYSTEM_PROMPT_ENGLISH

        user_prompt = self.prompts.build_user_prompt(
            query=query,
            context=context,
            language=language,
        )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Generate
        result = await self.llm.generate(full_prompt, stream=stream)

        # Parse citations if requested
        citations = []
        if self.include_citations:
            citations = self._parse_citations(result.answer, retrieved_chunks)

        # Add source information to result
        result.citations = citations

        return result

    def _parse_citations(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse citations from generated answer."""

        import re

        # Find citation patterns like [1], [2], etc.
        citation_pattern = r"\[(\d+)\]"
        citations_found = re.findall(citation_pattern, answer)

        citations = []
        for ref_num in set(citations_found):
            idx = int(ref_num) - 1
            if idx < len(chunks):
                chunk = chunks[idx]
                metadata = chunk.get("metadata", {})

                citations.append(
                    {
                        "reference": f"[{ref_num}]",
                        "book_title": metadata.get("book_title", "Unknown"),
                        "author": metadata.get("author", "Unknown"),
                        "category": metadata.get("category", "Unknown"),
                        "content_preview": chunk.get("content", "")[:200] + "...",
                    }
                )

        return citations

    def format_response(
        self,
        result: GenerationResult,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """Format response for API output."""

        response = {
            "answer": result.answer,
            "metadata": {
                "model": result.model,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
            },
        }

        if include_sources and result.citations:
            response["sources"] = result.citations

        return response


class ResponseGuardrails:
    """
    Safety checks for generated responses.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    async def validate(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate response before returning.

        Returns:
            Validation result with issues and confidence
        """

        issues = []

        # Check 1: Response length sanity
        if len(response) < 10:
            issues.append("Response too short")

        # Check 2: Potential hallucination (simple heuristic)
        if not self._is_grounded(response, retrieved_chunks):
            issues.append("Response may not be grounded in context")

        # Check 3: PII detection (basic)
        if self._contains_pii(response):
            issues.append("Response may contain PII")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "confidence": 1.0 - (len(issues) * 0.25),
        }

    def _is_grounded(
        self,
        response: str,
        chunks: List[Dict[str, Any]],
    ) -> bool:
        """Check if response appears to be grounded in context."""

        if not chunks:
            return False

        # Simple heuristic: check if response contains words from context
        context_text = " ".join([c.get("content", "") for c in chunks])

        # Check for key phrases from context
        response_lower = response.lower()

        # Extract some words from context
        context_words = set(context_text.lower().split())
        response_words = set(response_lower.split())

        # If less than 20% overlap, might be hallucinated
        if response_words and context_words:
            overlap = len(response_words & context_words) / len(response_words)
            return overlap > 0.2

        return True

    def _contains_pii(self, text: str) -> bool:
        """Basic PII detection."""

        import re

        # Email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

        # Phone pattern
        phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"

        return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
