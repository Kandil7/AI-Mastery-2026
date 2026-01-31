"""
Chat Enhancement Services
========================
Services for chat session management and AI-powered features.

خدمات تحسين المحادثة
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import logging

from src.domain.entities import TenantId

log = logging.getLogger(__name__)


@dataclass
class ChatSummary:
    """Summary of a chat session.

    ملخص جلسة المحادثة
    """

    summary: str
    topics: List[str]
    sentiment: str  # positive, neutral, negative
    question_count: int


class ChatTitleGenerator:
    """
    Generate intelligent titles for chat sessions.

    توليد عناوين ذكية لجلسات المحادثة
    """

    def __init__(self, llm_port):
        """
        Initialize with LLM port.

        Args:
            llm_port: LLM adapter for generation
        """
        self._llm = llm_port

    def generate_title(
        self,
        turns: List[dict[str, str]],
        max_length: int = 50,
    ) -> str:
        """
        Generate a concise title from chat turns.

        Args:
            turns: List of chat turns (question, answer)
            max_length: Maximum title length (default: 50)

        Returns:
            Generated title

        توليد عنوان مختصر من دورات المحادثة
        """
        if not turns:
            return "New Chat"

        # Get first question and last answer
        first_question = turns[0].get("question", "")[:100]
        last_answer = turns[-1].get("answer", "")[:100]

        # Build prompt for title generation
        prompt = f"""Generate a concise title (max {max_length} characters) for this chat session:
        
First question: {first_question}

Last answer: {last_answer}

Total turns: {len(turns)}

Guidelines:
- Use the main topic discussed
- Keep it short and descriptive
- Focus on what the user was asking about
- Do NOT include the word "Chat" or "Session"
- Provide ONLY the title, nothing else

Title:"""

        try:
            title = self._llm.generate(prompt, temperature=0.3)

            # Clean up and truncate
            title = title.strip()
            title = title.replace('"', "").replace("'", "")
            title = title[:max_length]

            return title

        except Exception as e:
            log.error("Failed to generate chat title", error=str(e))
            # Fallback: use first question
            return first_question[:max_length]

    def generate_title_async(self, turns: List[dict[str, str]], max_length: int = 50):
        """
        Async version of generate_title().

        نسخة غير متزامنة من generate_title()
        """
        import asyncio

        return asyncio.run(self.generate_title_async_impl(turns, max_length))

    async def generate_title_async_impl(self, turns, max_length):
        """Async implementation."""
        if not turns:
            return "New Chat"

        first_question = turns[0].get("question", "")[:100]
        last_answer = turns[-1].get("answer", "")[:100]

        prompt = f"""Generate a concise title (max {max_length} characters) for this chat session.

First question: {first_question}
Last answer: {last_answer}
Total turns: {len(turns)}

Provide ONLY the title, nothing else."""

        try:
            title = await self._llm.generate_async(prompt, temperature=0.3)
            title = title.strip()[:max_length]
            return title
        except Exception as e:
            log.error("Failed to generate async chat title", error=str(e))
            return first_question[:max_length]


class ChatSummarizer:
    """
    Generate summaries for completed chat sessions.

    توليد ملخصات لجلسات المحادثة المكتملة
    """

    def __init__(self, llm_port):
        """
        Initialize with LLM port.
        """
        self._llm = llm_port

    def summarize_session(
        self,
        turns: List[dict[str, str]],
        max_summary_length: int = 200,
    ) -> ChatSummary:
        """
        Generate a summary of the chat session.

        Args:
            turns: List of chat turns
            max_summary_length: Maximum summary length

        Returns:
            ChatSummary with summary, topics, sentiment

        توليد ملخص لجلسة المحادثة
        """
        if not turns:
            return ChatSummary(
                summary="Empty session",
                topics=[],
                sentiment="neutral",
                question_count=0,
            )

        # Concatenate Q&A for context
        qa_text = "\n\n".join(
            [
                f"Q{i + 1}: {turn.get('question', '')[:200]}"
                f"A{i + 1}: {turn.get('answer', '')[:300]}"
                for i, turn in enumerate(turns[-5:])  # Last 5 turns max
            ]
        )

        # Build prompt for summarization
        prompt = f"""Analyze this chat session and provide:

1. A concise summary ({max_summary_length} chars max)
2. Main topics discussed (comma-separated, max 5)
3. Overall sentiment (positive/neutral/negative)

Chat Session:
{qa_text}

Format your response as:
Summary: [summary]
Topics: [topic1, topic2, ...]
Sentiment: [positive/neutral/negative]"""

        try:
            response = self._llm.generate(prompt, temperature=0.2)

            # Parse response
            summary = self._parse_summary_response(response)

            summary.question_count = len(turns)
            return summary

        except Exception as e:
            log.error("Failed to summarize chat session", error=str(e))
            return ChatSummary(
                summary="Summary generation failed",
                topics=[],
                sentiment="neutral",
                question_count=len(turns),
            )

    def _parse_summary_response(self, response: str) -> ChatSummary:
        """Parse LLM response for summary components."""
        try:
            summary = ""
            topics = []
            sentiment = "neutral"

            for line in response.split("\n"):
                line = line.strip()
                if line.lower().startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.lower().startswith("topics:"):
                    topics_str = line.split(":", 1)[1].strip()
                    topics = [t.strip() for t in topics_str.split(",")][:5]
                elif line.lower().startswith("sentiment:"):
                    sentiment = line.split(":", 1)[1].strip().lower()

            # Default values if parsing failed
            summary = summary or "Unable to generate summary"
            topics = topics or ["General discussion"]
            sentiment = sentiment if sentiment in ["positive", "neutral", "negative"] else "neutral"

            return ChatSummary(
                summary=summary,
                topics=topics,
                sentiment=sentiment,
                question_count=0,  # Will be set by caller
            )

        except Exception as e:
            log.error("Failed to parse summary response", error=str(e))
            return ChatSummary(
                summary=response[:200],  # Use raw response as fallback
                topics=["General"],
                sentiment="neutral",
                question_count=0,
            )


class SessionManager:
    """
    Manages chat sessions with title generation and summarization.

    يدير جلسات المحادثة مع توليد العناوين والملخصات
    """

    def __init__(
        self,
        title_generator: ChatTitleGenerator,
        summarizer: ChatSummarizer,
        chat_repo,
    ):
        """
        Initialize session manager.

        Args:
            title_generator: Title generation service
            summarizer: Chat summarization service
            chat_repo: Chat repository for persistence
        """
        self._title_gen = title_generator
        self._summarizer = summarizer
        self._chat_repo = chat_repo

    def auto_title_session(
        self,
        session_id: str,
        turns: List[dict[str, str]],
        save_to_db: bool = False,
        tenant_id: TenantId = None,
    ) -> str:
        """
        Automatically generate and save title for a session.

        Args:
            session_id: Chat session ID
            turns: Chat turns for title generation
            save_to_db: Whether to save to database (default: False)
            tenant_id: Owner tenant ID

        Returns:
            Generated title

        توليد وحفظ عنوان تلقائي لجلسة
        """
        title = self._title_gen.generate_title(turns)

        if save_to_db and self._chat_repo and tenant_id:
            self._chat_repo.update_session_title(
                tenant_id=tenant_id,
                session_id=session_id,
                title=title,
            )
            log.info("Session title generated and saved", session_id=session_id, title=title)

        return title

    def summarize_and_close_session(
        self,
        session_id: str,
        turns: List[dict[str, str]],
        tenant_id: TenantId = None,
    ) -> ChatSummary:
        """
        Summarize and close a chat session.

        Args:
            session_id: Chat session ID
            turns: All chat turns in session
            tenant_id: Owner tenant ID

        Returns:
            Session summary

        تلخيص وإغلاق جلسة المحادثة
        """
        summary = self._summarizer.summarize_session(turns)

        if self._chat_repo and tenant_id:
            self._chat_repo.update_session_summary(
                tenant_id=tenant_id,
                session_id=session_id,
                summary=summary.summary,
                topics=summary.topics,
                sentiment=summary.sentiment,
            )

        log.info(
            "Session summarized",
            session_id=session_id,
            summary_length=len(summary.summary),
            topics=summary.topics,
            sentiment=summary.sentiment,
        )

        return summary


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_session_turns(
    session_id: str,
    max_turns: int = 20,
) -> List[dict[str, str]]:
    """
    Get chat turns for a session (placeholder for database integration).

    الحصول على دورات المحادثة لجلسة
    """
    # TODO: Implement database query
    # return db.query(ChatTurn).filter_by(session_id=session_id).all()
    return []


def create_sample_session() -> List[dict[str, str]]:
    """Create sample session for testing."""
    return [
        {"question": "What is RAG?", "answer": "RAG stands for Retrieval-Augmented Generation..."},
        {"question": "How does vector search work?", "answer": "Vector search uses embeddings..."},
        {"question": "What are the benefits?", "answer": "RAG improves accuracy..."},
    ]


if __name__ == "__main__":
    # Test chat enhancements
    from unittest.mock import Mock

    llm_mock = Mock()
    llm_mock.generate.return_value = "RAG Architecture Discussion"
    llm_mock.generate_async.return_value = "Session Summary: Discussion of RAG system architecture"

    title_gen = ChatTitleGenerator(llm_mock)
    summarizer = ChatSummarizer(llm_mock)

    # Test title generation
    turns = create_sample_session()
    title = title_gen.generate_title(turns)
    print(f"Generated Title: {title}")

    # Test summarization
    summary = summarizer.summarize_session(turns)
    print(f"Summary: {summary.summary}")
    print(f"Topics: {summary.topics}")
    print(f"Sentiment: {summary.sentiment}")

    # Test session manager
    manager = SessionManager(title_gen, summarizer)
    auto_title = manager.auto_title_session("session-123", turns)
    print(f"Auto-generated Title: {auto_title}")
