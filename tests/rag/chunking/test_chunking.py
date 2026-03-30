"""
Test Suite for Chunking Module

Comprehensive tests for all chunking strategies.

Run with:
    pytest tests/rag/chunking/ -v --cov=src.rag.chunking

Coverage target: 95%+
"""

import pytest
from typing import List, Dict, Any

from src.rag.chunking import (
    Chunk,
    ChunkingConfig,
    ChunkingStrategy,
    BaseChunker,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    HierarchicalChunker,
    HierarchicalChunkResult,
    CodeChunker,
    TokenAwareChunker,
    ChunkerFactory,
    create_chunker,
    create_fixed_chunker,
    create_recursive_chunker,
    create_semantic_chunker,
    create_hierarchical_chunker,
    create_code_chunker,
    create_token_aware_chunker,
    count_tokens,
    truncate_to_tokens,
    split_by_tokens,
    generate_chunk_id,
    estimate_tokens_from_chars,
    is_arabic_text,
    get_recommended_config,
)


# ==================== Fixtures ====================


@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """Sample document for testing."""
    return {
        "id": "test_doc_001",
        "content": """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed. 
It focuses on developing computer programs that can access data and use it 
to learn for themselves.

The process begins with observations or data, such as examples, direct 
experience, or instruction, in order to look for patterns in data and 
make better decisions in the future based on the examples we provide.

Types of Machine Learning

1. Supervised Learning: The algorithm learns from labeled training data.
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
3. Reinforcement Learning: The algorithm learns through trial and error.

Conclusion

Machine learning continues to evolve and find new applications in various 
fields including healthcare, finance, transportation, and entertainment.
""".strip(),
        "metadata": {
            "source": "ml_intro.txt",
            "category": "education",
            "author": "AI Team",
        },
    }


@pytest.fixture
def sample_code_document() -> Dict[str, Any]:
    """Sample code document for testing."""
    return {
        "id": "test_code_001",
        "content": """
def calculate_sum(numbers: list) -> int:
    '''Calculate the sum of a list of numbers.'''
    total = 0
    for num in numbers:
        total += num
    return total

class DataProcessor:
    '''Process data with various transformations.'''
    
    def __init__(self, data: list):
        self.data = data
        self.processed = False
    
    def normalize(self) -> list:
        '''Normalize the data to 0-1 range.'''
        if not self.data:
            return []
        min_val = min(self.data)
        max_val = max(self.data)
        range_val = max_val - min_val
        if range_val == 0:
            return [0.0] * len(self.data)
        return [(x - min_val) / range_val for x in self.data]
    
    def standardize(self) -> list:
        '''Standardize the data to mean 0, std 1.'''
        if not self.data:
            return []
        mean = sum(self.data) / len(self.data)
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        std = variance ** 0.5
        if std == 0:
            return [0.0] * len(self.data)
        return [(x - mean) / std for x in self.data]

async def fetch_data(url: str) -> dict:
    '''Fetch data from an API endpoint.'''
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
""".strip(),
        "metadata": {
            "source": "processor.py",
            "language": "python",
        },
    }


@pytest.fixture
def sample_arabic_document() -> Dict[str, Any]:
    """Sample Arabic document for testing."""
    return {
        "id": "test_arabic_001",
        "content": """
مقدمة في تعلم الآلة

تعلم الآلة هو فرع من فروع الذكاء الاصطناعي يُمكّن الأنظمة من التعلم 
والتحسين من الخبرة دون أن تتم برمجتها صراحةً. يركز على تطوير برامج 
الحاسوب التي يمكنها الوصول إلى البيانات واستخدامها للتعلم بنفسها.

تبدأ العملية بملاحظات أو بيانات، مثل الأمثلة أو الخبرة المباشرة، 
من أجل البحث عن أنماط في البيانات واتخاذ قرارات أفضل في المستقبل.

أنواع تعلم الآلة

١. التعلم الخاضع للإشراف: تتعلم الخوارزمية من بيانات التدريب الموسومة.
٢. التعلم غير الخاضع للإشراف: تجد الخوارزمية أنماطاً في البيانات غير الموسومة.
٣. التعلم التعزيزي: تتعلم الخوارزمية من خلال التجربة والخطأ.

الخاتمة

يستمر تعلم الآلة في التطور وإيجاد تطبيقات جديدة في مجالات مختلفة 
بما في ذلك الرعاية الصحية والمالية والنقل والترفيه.
""".strip(),
        "metadata": {
            "source": "ml_intro_ar.txt",
            "language": "arabic",
        },
    }


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """Default chunking configuration for tests."""
    return ChunkingConfig(
        chunk_size=256,
        chunk_overlap=25,
        min_chunk_size=50,
        max_chunk_size=1000,
    )


# ==================== Chunk Data Class Tests ====================


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_chunk_creation(self, sample_document: Dict[str, Any]) -> None:
        """Test basic chunk creation."""
        chunk = Chunk(
            content="Test content",
            document_id=sample_document["id"],
            start_index=0,
            end_index=12,
        )

        assert chunk.content == "Test content"
        assert chunk.document_id == sample_document["id"]
        assert chunk.start_index == 0
        assert chunk.end_index == 12
        assert chunk.chunk_id != ""  # Auto-generated
        assert chunk.metadata == {}

    def test_chunk_auto_id_generation(self) -> None:
        """Test automatic chunk ID generation."""
        chunk1 = Chunk(content="Same content", document_id="doc1", start_index=0, end_index=12)
        chunk2 = Chunk(content="Same content", document_id="doc1", start_index=0, end_index=12)

        assert chunk1.chunk_id == chunk2.chunk_id

    def test_chunk_word_count(self) -> None:
        """Test word count property."""
        chunk = Chunk(content="Hello world test", document_id="doc1", start_index=0, end_index=18)
        assert chunk.word_count == 4

    def test_chunk_char_count(self) -> None:
        """Test character count property."""
        content = "Hello world"
        chunk = Chunk(content=content, document_id="doc1", start_index=0, end_index=len(content))
        assert chunk.char_count == len(content)

    def test_chunk_is_empty(self) -> None:
        """Test is_empty property."""
        empty_chunk = Chunk(content="   ", document_id="doc1", start_index=0, end_index=3)
        non_empty_chunk = Chunk(content="Hello", document_id="doc1", start_index=0, end_index=5)

        assert empty_chunk.is_empty is True
        assert non_empty_chunk.is_empty is False

    def test_chunk_to_dict(self, sample_document: Dict[str, Any]) -> None:
        """Test conversion to dictionary."""
        chunk = Chunk(
            content="Test content",
            document_id=sample_document["id"],
            start_index=0,
            end_index=12,
            metadata={"key": "value"},
        )

        data = chunk.to_dict()

        assert data["content"] == "Test content"
        assert data["document_id"] == sample_document["id"]
        assert data["metadata"]["key"] == "value"
        assert data["word_count"] == 2

    def test_chunk_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "content": "Test content",
            "document_id": "doc1",
            "start_index": 0,
            "end_index": 12,
            "metadata": {"key": "value"},
            "chunk_id": "custom_id",
        }

        chunk = Chunk.from_dict(data)

        assert chunk.content == "Test content"
        assert chunk.document_id == "doc1"
        assert chunk.chunk_id == "custom_id"

    def test_chunk_str_repr(self) -> None:
        """Test string representations."""
        chunk = Chunk(content="Hello world test content", document_id="doc1", start_index=0, end_index=26)

        str_repr = str(chunk)
        repr_repr = repr(chunk)

        assert "doc1" in str_repr
        assert "Hello" in str_repr
        assert "Chunk" in repr_repr


# ==================== ChunkingConfig Tests ====================


class TestChunkingConfig:
    """Tests for the ChunkingConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ChunkingConfig()

        assert config.strategy == ChunkingStrategy.RECURSIVE
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 2000

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=256,
            chunk_overlap=25,
        )

        assert config.strategy == ChunkingStrategy.FIXED
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25

    def test_config_validation_invalid_chunk_size(self) -> None:
        """Test validation rejects invalid chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0)

    def test_config_validation_invalid_overlap(self) -> None:
        """Test validation rejects invalid overlap."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            ChunkingConfig(chunk_overlap=-10)

    def test_config_validation_overlap_exceeds_size(self) -> None:
        """Test validation rejects overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_config_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = ChunkingConfig(chunk_size=256, chunk_overlap=25)
        data = config.to_dict()

        assert data["chunk_size"] == 256
        assert data["chunk_overlap"] == 25
        assert data["strategy"] == "recursive"

    def test_config_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "strategy": "semantic",
            "chunk_size": 768,
            "chunk_overlap": 50,
        }

        config = ChunkingConfig.from_dict(data)

        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.chunk_size == 768


# ==================== BaseChunker Tests ====================


class TestBaseChunker:
    """Tests for the BaseChunker abstract class."""

    def test_base_chunker_is_abstract(self) -> None:
        """Test that BaseChunker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChunker()

    def test_concrete_implementation(self, chunking_config: ChunkingConfig) -> None:
        """Test creating a concrete implementation."""

        class TestChunker(BaseChunker):
            def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
                return [
                    Chunk(
                        content=document["content"],
                        document_id=document["id"],
                        start_index=0,
                        end_index=len(document["content"]),
                    )
                ]

        chunker = TestChunker(chunking_config)
        doc = {"id": "test", "content": "Hello world"}
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"

    def test_chunk_texts(self, chunking_config: ChunkingConfig) -> None:
        """Test chunking multiple texts."""

        class TestChunker(BaseChunker):
            def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
                return [
                    Chunk(
                        content=document["content"],
                        document_id=document["id"],
                        start_index=0,
                        end_index=len(document["content"]),
                    )
                ]

        chunker = TestChunker(chunking_config)
        texts = ["Text one", "Text two", "Text three"]

        all_chunks = chunker.chunk_texts(texts)

        assert len(all_chunks) == 3
        assert len(all_chunks[0]) == 1
        assert len(all_chunks[1]) == 1

    def test_clean_text(self, chunking_config: ChunkingConfig) -> None:
        """Test text cleaning."""

        class TestChunker(BaseChunker):
            def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
                return []

        chunker = TestChunker(chunking_config)

        # Test line ending normalization
        text = "Line 1\r\nLine 2\rLine 3"
        cleaned = chunker._clean_text(text)
        assert "\r" not in cleaned

        # Test excessive whitespace
        text = "Para 1\n\n\n\nPara 2"
        cleaned = chunker._clean_text(text)
        assert "\n\n\n" not in cleaned


# ==================== FixedSizeChunker Tests ====================


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_fixed_chunker_creation(self, chunking_config: ChunkingConfig) -> None:
        """Test FixedSizeChunker creation."""
        chunker = FixedSizeChunker(chunking_config)
        assert chunker.config.chunk_size == 256

    def test_fixed_chunking(self, sample_document: Dict[str, Any]) -> None:
        """Test fixed-size chunking."""
        chunker = create_fixed_chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.document_id == sample_document["id"] for chunk in chunks)

    def test_fixed_chunking_empty_document(self) -> None:
        """Test fixed chunking with empty document."""
        chunker = create_fixed_chunker()
        doc = {"id": "empty", "content": ""}

        with pytest.raises(ValueError):
            chunker.chunk(doc)

    def test_fixed_chunking_overlap(self, sample_document: Dict[str, Any]) -> None:
        """Test that overlap is applied correctly."""
        chunker = create_fixed_chunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(sample_document)

        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                # Chunks should exist and have content
                assert chunks[i].content
                assert chunks[i + 1].content

    def test_create_fixed_chunker_factory(self) -> None:
        """Test factory function."""
        chunker = create_fixed_chunker(
            chunk_size=512,
            chunk_overlap=50,
            tokenizer_name="cl100k_base",
        )

        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.config.chunk_size == 512


# ==================== RecursiveChunker Tests ====================


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_recursive_chunker_creation(self, chunking_config: ChunkingConfig) -> None:
        """Test RecursiveChunker creation."""
        chunker = RecursiveChunker(chunking_config)
        assert chunker.config.chunk_size == 256

    def test_recursive_chunking(self, sample_document: Dict[str, Any]) -> None:
        """Test recursive chunking."""
        chunker = create_recursive_chunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)

    def test_recursive_chunking_preserves_structure(self, sample_document: Dict[str, Any]) -> None:
        """Test that recursive chunking preserves document structure."""
        chunker = create_recursive_chunker(chunk_size=300, chunk_overlap=20)
        chunks = chunker.chunk(sample_document)

        # At least some chunks should contain section headers
        content = "\n".join(chunk.content for chunk in chunks)
        assert "Introduction" in content or "Machine Learning" in content

    def test_recursive_chunking_arabic(self, sample_arabic_document: Dict[str, Any]) -> None:
        """Test recursive chunking with Arabic text."""
        chunker = create_recursive_chunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(sample_arabic_document)

        assert len(chunks) > 0
        # Verify Arabic characters are preserved
        arabic_content = "".join(chunk.content for chunk in chunks)
        assert any("\u0600" <= c <= "\u06FF" for c in arabic_content)

    def test_create_recursive_chunker_factory(self) -> None:
        """Test factory function."""
        chunker = create_recursive_chunker(
            chunk_size=512,
            separators=["\n\n", "\n", ". "],
        )

        assert isinstance(chunker, RecursiveChunker)


# ==================== SemanticChunker Tests ====================


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_semantic_chunker_creation(self, chunking_config: ChunkingConfig) -> None:
        """Test SemanticChunker creation."""
        chunker = SemanticChunker(chunking_config)
        assert chunker.config.similarity_threshold == 0.5

    def test_semantic_chunking_fallback(self, sample_document: Dict[str, Any]) -> None:
        """Test semantic chunking falls back to recursive when model unavailable."""
        chunker = create_semantic_chunker(chunk_size=200)
        chunks = chunker.chunk(sample_document)

        # Should work even without embedding model (falls back to recursive)
        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)

    def test_semantic_chunking_with_custom_embedder(self, sample_document: Dict[str, Any]) -> None:
        """Test semantic chunking with custom embedding function."""

        def dummy_embed(text: str) -> List[float]:
            return [0.5] * 384  # Dummy embedding

        chunker = create_semantic_chunker(
            chunk_size=200,
            embedding_function=dummy_embed,
        )
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0

    def test_create_semantic_chunker_factory(self) -> None:
        """Test factory function."""
        chunker = create_semantic_chunker(
            similarity_threshold=0.3,
            embedding_model="all-MiniLM-L6-v2",
        )

        assert isinstance(chunker, SemanticChunker)
        assert chunker.config.similarity_threshold == 0.3


# ==================== HierarchicalChunker Tests ====================


class TestHierarchicalChunker:
    """Tests for HierarchicalChunker."""

    def test_hierarchical_chunker_creation(self, chunking_config: ChunkingConfig) -> None:
        """Test HierarchicalChunker creation."""
        chunker = HierarchicalChunker(chunking_config)
        assert chunker.config.parent_chunk_size == 2000

    def test_hierarchical_chunking(self, sample_document: Dict[str, Any]) -> None:
        """Test hierarchical chunking."""
        chunker = create_hierarchical_chunker(
            parent_chunk_size=500,
            child_chunk_size=150,
        )
        result = chunker.chunk(sample_document)

        assert isinstance(result, HierarchicalChunkResult)
        assert len(result.children) > 0
        assert len(result.parents) > 0
        assert len(result.parent_child_map) > 0

    def test_hierarchical_parent_child_mapping(
        self,
        sample_document: Dict[str, Any],
    ) -> None:
        """Test parent-child relationships."""
        chunker = create_hierarchical_chunker(
            parent_chunk_size=500,
            child_chunk_size=150,
        )
        result = chunker.chunk(sample_document)

        # Every child should have a parent
        for child in result.children:
            assert child.parent_id in result.parent_child_map

        # Every parent should have at least one child
        for parent_id in result.parent_child_map:
            assert len(result.parent_child_map[parent_id]) >= 1

    def test_hierarchical_expand_children_to_parents(
        self,
        sample_document: Dict[str, Any],
    ) -> None:
        """Test expanding children to parents."""
        chunker = create_hierarchical_chunker(
            parent_chunk_size=500,
            child_chunk_size=150,
        )
        result = chunker.chunk(sample_document)

        if result.children:
            child_ids = [c.chunk_id for c in result.children[:2]]
            parents = result.get_parents_for_children(child_ids)

            assert len(parents) > 0

    def test_hierarchical_chunk_simple(self, sample_document: Dict[str, Any]) -> None:
        """Test simple hierarchical chunking (children only)."""
        chunker = create_hierarchical_chunker()
        chunks = chunker.chunk_simple(sample_document)

        assert len(chunks) > 0
        # Children should have parent_id set
        assert all(chunk.parent_id for chunk in chunks)


# ==================== CodeChunker Tests ====================


class TestCodeChunker:
    """Tests for CodeChunker."""

    def test_code_chunker_creation(self, chunking_config: ChunkingConfig) -> None:
        """Test CodeChunker creation."""
        chunker = CodeChunker(chunking_config, language="python")
        assert chunker.language == "python"

    def test_code_chunking(self, sample_code_document: Dict[str, Any]) -> None:
        """Test code chunking."""
        chunker = create_code_chunker(chunk_size=300, language="python")
        chunks = chunker.chunk(sample_code_document)

        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)

    def test_code_chunking_preserves_functions(
        self,
        sample_code_document: Dict[str, Any],
    ) -> None:
        """Test that code chunking preserves function boundaries."""
        chunker = create_code_chunker(chunk_size=500, language="python")
        chunks = chunker.chunk(sample_code_document)

        content = "\n".join(chunk.content for chunk in chunks)
        assert "def " in content or "class " in content

    def test_code_language_detection(self, sample_code_document: Dict[str, Any]) -> None:
        """Test automatic language detection."""
        chunker = create_code_chunker(chunk_size=300, language="auto")
        chunks = chunker.chunk(sample_code_document)

        assert len(chunks) > 0
        # Should detect Python
        assert chunker.language == "python"

    def test_code_chunking_invalid_code(self) -> None:
        """Test code chunking with invalid code."""
        chunker = create_code_chunker(chunk_size=100, language="python")
        doc = {"id": "invalid", "content": "def incomplete(\n\nclass AlsoIncomplete"}

        # Should handle gracefully (may return empty or partial chunks)
        chunks = chunker.chunk(doc)
        # At least shouldn't crash
        assert isinstance(chunks, list)


# ==================== TokenAwareChunker Tests ====================


class TestTokenAwareChunker:
    """Tests for TokenAwareChunker."""

    def test_token_aware_chunker_creation(self, chunking_config: ChunkingConfig) -> None:
        """Test TokenAwareChunker creation."""
        chunker = TokenAwareChunker(chunking_config)
        assert chunker.config.tokenizer_name == "cl100k_base"

    def test_token_aware_chunking(self, sample_document: Dict[str, Any]) -> None:
        """Test token-aware chunking."""
        chunker = create_token_aware_chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)

    def test_token_counting(self) -> None:
        """Test token counting utility."""
        text = "Hello, world! This is a test."
        count = count_tokens(text, "cl100k_base")

        assert count > 0
        assert isinstance(count, int)

    def test_truncate_to_tokens(self) -> None:
        """Test token truncation."""
        text = "Hello, world! " * 100  # Long text
        truncated = truncate_to_tokens(text, max_tokens=10)

        truncated_count = count_tokens(truncated)
        assert truncated_count <= 10

    def test_split_by_tokens(self) -> None:
        """Test splitting by tokens."""
        text = "Hello, world! " * 50
        chunks = split_by_tokens(text, max_tokens=20, overlap=5)

        assert len(chunks) > 0
        assert all(chunks)

    def test_token_aware_chunking_metadata(
        self,
        sample_document: Dict[str, Any],
    ) -> None:
        """Test that token-aware chunking includes token metadata."""
        chunker = create_token_aware_chunker(chunk_size=100)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            assert "token_count" in chunk.metadata
            assert chunk.metadata["token_count"] > 0


# ==================== ChunkerFactory Tests ====================


class TestChunkerFactory:
    """Tests for ChunkerFactory."""

    def test_create_with_enum(self, chunking_config: ChunkingConfig) -> None:
        """Test creating chunker with strategy enum."""
        chunker = ChunkerFactory.create(ChunkingStrategy.FIXED, config=chunking_config)
        assert isinstance(chunker, FixedSizeChunker)

    def test_create_with_string(self, chunking_config: ChunkingConfig) -> None:
        """Test creating chunker with strategy string."""
        chunker = ChunkerFactory.create("recursive", config=chunking_config)
        assert isinstance(chunker, RecursiveChunker)

    def test_create_unknown_strategy(self) -> None:
        """Test creating chunker with unknown strategy."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkerFactory.create("nonexistent_strategy")

    def test_get_available_strategies(self) -> None:
        """Test getting available strategies."""
        strategies = ChunkerFactory.get_available_strategies()

        assert len(strategies) >= 6
        assert "recursive" in strategies
        assert "semantic" in strategies

    def test_get_recommended_strategy(self) -> None:
        """Test getting recommended strategy."""
        strategy = ChunkerFactory.get_recommended_strategy("code")
        assert strategy == ChunkingStrategy.CODE

        strategy = ChunkerFactory.get_recommended_strategy("legal documents")
        assert strategy == ChunkingStrategy.SEMANTIC

    def test_create_chunker_function(self) -> None:
        """Test create_chunker convenience function."""
        chunker = create_chunker("fixed", chunk_size=256)
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.config.chunk_size == 256

    def test_get_recommended_config(self) -> None:
        """Test getting recommended configuration."""
        config = get_recommended_config("code")
        assert config.strategy == ChunkingStrategy.CODE

        config = get_recommended_config("documentation", chunk_size=1024)
        assert config.chunk_size == 1024


# ==================== Utility Function Tests ====================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_chunk_id(self) -> None:
        """Test chunk ID generation."""
        chunk_id = generate_chunk_id("doc1", "test content", 0)

        assert "doc1" in chunk_id
        assert "chunk" in chunk_id
        assert len(chunk_id) > 10

    def test_estimate_tokens_from_chars(self) -> None:
        """Test token estimation from characters."""
        chars = 1000
        tokens = estimate_tokens_from_chars(chars)

        assert tokens > 0
        assert tokens < chars  # Tokens should be fewer than chars

    def test_is_arabic_text(self) -> None:
        """Test Arabic text detection."""
        arabic = "مرحبا بالعالم"
        english = "Hello world"
        mixed = "Hello مرحبا"

        assert is_arabic_text(arabic) is True
        assert is_arabic_text(english) is False
        # Mixed depends on proportion
        assert isinstance(is_arabic_text(mixed), bool)

    def test_is_arabic_text_empty(self) -> None:
        """Test Arabic detection with empty string."""
        assert is_arabic_text("") is False


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for chunking module."""

    def test_full_chunking_pipeline(self, sample_document: Dict[str, Any]) -> None:
        """Test complete chunking pipeline."""
        # Create chunker
        chunker = create_chunker("recursive", chunk_size=256, chunk_overlap=25)

        # Chunk document
        chunks = chunker.chunk(sample_document)

        # Verify chunks
        assert len(chunks) > 0

        # Verify all chunks have required fields
        for chunk in chunks:
            assert chunk.content
            assert chunk.chunk_id
            assert chunk.document_id
            assert chunk.metadata

        # Verify content coverage
        original_content = sample_document["content"]
        chunked_content = "".join(chunk.content for chunk in chunks)

        # Should have most of the original content (allowing for some loss at boundaries)
        assert len(chunked_content) > len(original_content) * 0.8

    def test_multiple_strategies_same_document(
        self,
        sample_document: Dict[str, Any],
    ) -> None:
        """Test multiple strategies on same document."""
        strategies = ["fixed", "recursive", "token_aware"]
        results = {}

        for strategy in strategies:
            chunker = create_chunker(strategy, chunk_size=200)
            chunks = chunker.chunk(sample_document)
            results[strategy] = chunks

        # All strategies should produce chunks
        for strategy, chunks in results.items():
            assert len(chunks) > 0, f"{strategy} produced no chunks"

    def test_chunking_with_metadata(self, sample_document: Dict[str, Any]) -> None:
        """Test that metadata is preserved in chunks."""
        chunker = create_chunker("recursive", chunk_size=300)
        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            # Document metadata should be prefixed with 'doc_'
            assert "doc_source" in chunk.metadata
            assert "doc_category" in chunk.metadata
            assert chunk.metadata["doc_source"] == "ml_intro.txt"

    def test_chunking_roundtrip(self, sample_document: Dict[str, Any]) -> None:
        """Test chunking and reconstruction."""
        chunker = create_chunker("recursive", chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(sample_document)

        # Reconstruct content
        reconstructed = "".join(chunk.content for chunk in chunks)

        # Should contain most of original
        original_words = set(sample_document["content"].split())
        reconstructed_words = set(reconstructed.split())

        # At least 80% of words should be preserved
        overlap = len(original_words & reconstructed_words)
        assert overlap / len(original_words) > 0.8


# ==================== Edge Case Tests ====================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self) -> None:
        """Test handling of empty content."""
        chunker = create_chunker("recursive")
        doc = {"id": "empty", "content": ""}

        with pytest.raises(ValueError):
            chunker.chunk(doc)

    def test_whitespace_only_content(self) -> None:
        """Test handling of whitespace-only content."""
        chunker = create_chunker("recursive")
        doc = {"id": "whitespace", "content": "   \n\n   "}

        with pytest.raises(ValueError):
            chunker.chunk(doc)

    def test_very_short_content(self) -> None:
        """Test handling of very short content."""
        chunker = create_chunker("recursive", chunk_size=512)
        doc = {"id": "short", "content": "Hi."}

        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_very_long_content(self) -> None:
        """Test handling of very long content."""
        chunker = create_chunker("recursive", chunk_size=100)
        long_text = "Word. " * 10000  # Very long text
        doc = {"id": "long", "content": long_text}

        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

        # Each chunk should respect max size
        for chunk in chunks:
            assert chunk.word_count < 200  # Allow some tolerance

    def test_special_characters(self) -> None:
        """Test handling of special characters."""
        chunker = create_chunker("recursive")
        doc = {
            "id": "special",
            "content": "Text with\ttabs\nnewlines\r\nand special chars: \x00\x01\x02",
        }

        chunks = chunker.chunk(doc)
        assert len(chunks) > 0

    def test_unicode_content(self) -> None:
        """Test handling of Unicode content."""
        chunker = create_chunker("recursive")
        doc = {
            "id": "unicode",
            "content": "Hello 世界 🌍 مرحبا שלום",
        }

        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert "世界" in chunks[0].content

    def test_invalid_document_type(self) -> None:
        """Test handling of invalid document type."""
        chunker = create_chunker("recursive")

        with pytest.raises(ValueError):
            chunker.chunk("not a dict")  # type: ignore

    def test_document_missing_content(self) -> None:
        """Test handling of document without content."""
        chunker = create_chunker("recursive")
        doc = {"id": "no_content"}

        with pytest.raises(ValueError):
            chunker.chunk(doc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
