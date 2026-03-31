# Lab 1: Modality Detection and Classification

## 🎯 Lab Objectives

By completing this lab, you will:
1. Implement a query modality classifier
2. Build content modality detection for indexing
3. Create embedding-based modality verification
4. Evaluate detection accuracy on test data

## 📋 Prerequisites

- Python 3.10+
- Required packages installed (see Module README)
- Understanding of modality types (text, image, code, table)

## ⏱️ Time Estimate: 3-4 hours

---

## Part 1: Query Modality Classifier

### Task 1.1: Implement Pattern-Based Classification

Create a classifier that detects expected modality from query patterns:

```python
# File: lab_1_modality_detection/solution.py

from enum import Enum
from typing import List, Dict
from dataclasses import dataclass

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    TABLE = "table"
    MIXED = "mixed"

@dataclass
class ModalityPrediction:
    primary: ModalityType
    confidence: float
    secondary: List[ModalityType]
    reasoning: str

class QueryModalityClassifier:
    """
    Classifies queries by expected response modality.
    """
    
    IMAGE_INDICATORS = [
        "diagram", "chart", "graph", "image", "picture", "photo",
        "screenshot", "visual", "illustration", "architecture",
        "flow", "map", "schema", "blueprint", "drawing", "figure"
    ]
    
    CODE_INDICATORS = [
        "code", "snippet", "function", "method", "class",
        "implementation", "example", "script", "query",
        "api", "endpoint", "algorithm", "pattern", "def ",
        "import ", "function(", "class "
    ]
    
    TABLE_INDICATORS = [
        "table", "spreadsheet", "data", "metrics", "statistics",
        "comparison", "matrix", "grid", "specifications",
        "parameters", "configuration", "settings", "csv",
        "columns", "rows", "values"
    ]
    
    def classify(self, query: str) -> ModalityPrediction:
        """
        Classify query by detecting modality indicators.
        
        Args:
            query: User query string
            
        Returns:
            ModalityPrediction with primary modality and confidence
        """
        query_lower = query.lower()
        
        # Count indicators for each modality
        image_score = sum(1 for ind in self.IMAGE_INDICATORS if ind in query_lower)
        code_score = sum(1 for ind in self.CODE_INDICATORS if ind in query_lower)
        table_score = sum(1 for ind in self.TABLE_INDICATORS if ind in query_lower)
        
        # Text is default with baseline score
        scores = {
            ModalityType.IMAGE: image_score,
            ModalityType.CODE: code_score,
            ModalityType.TABLE: table_score,
            ModalityType.TEXT: max(1, 5 - image_score - code_score - table_score)
        }
        
        # Determine primary modality
        primary = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[primary] / total if total > 0 else 0.25
        
        # Get secondary modalities (those with score > 0)
        sorted_modalities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary = [m for m, s in sorted_modalities[1:] if s > 0]
        
        # Generate reasoning
        indicator_map = {
            ModalityType.IMAGE: self.IMAGE_INDICATORS,
            ModalityType.CODE: self.CODE_INDICATORS,
            ModalityType.TABLE: self.TABLE_INDICATORS
        }
        
        found_indicators = [
            ind for ind in indicator_map.get(primary, [])
            if ind in query_lower
        ][:3]  # Top 3 indicators
        
        reasoning = f"Detected {scores[primary]} indicators for {primary.value}: {', '.join(found_indicators)}"
        
        return ModalityPrediction(
            primary=primary,
            confidence=round(confidence, 3),
            secondary=secondary,
            reasoning=reasoning
        )
```

### Task 1.2: Test Your Classifier

```python
def test_classifier():
    classifier = QueryModalityClassifier()
    
    test_queries = [
        ("Show me the architecture diagram", ModalityType.IMAGE),
        ("Write a function to calculate sum", ModalityType.CODE),
        ("Display the revenue data in a table", ModalityType.TABLE),
        ("Explain how the system works", ModalityType.TEXT),
        ("Show me the code and the flow chart", ModalityType.MIXED),
    ]
    
    for query, expected in test_queries:
        result = classifier.classify(query)
        print(f"Query: {query}")
        print(f"Predicted: {result.primary.value} (confidence: {result.confidence})")
        print(f"Expected: {expected.value}")
        print(f"Match: {result.primary == expected}")
        print("-" * 50)

if __name__ == "__main__":
    test_classifier()
```

---

## Part 2: Content Modality Detection

### Task 2.1: File-Based Detection

```python
import mimetypes
from pathlib import Path
from PIL import Image

class ContentModalityDetector:
    """
    Detects modality of content for proper indexing.
    """
    
    def __init__(self):
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
        self.code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.go', '.rs', '.rb', '.sql'}
        self.table_extensions = {'.csv', '.xlsx', '.xls', '.tsv', '.parquet'}
    
    def detect_from_file(self, file_path: str) -> ModalityType:
        """Detect modality from file path and content."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        # Check extension first
        if ext in self.image_extensions:
            return ModalityType.IMAGE
        elif ext in self.code_extensions:
            return ModalityType.CODE
        elif ext in self.table_extensions:
            return ModalityType.TABLE
        
        # For text files, analyze content
        return self._detect_text_modality(path)
    
    def _detect_text_modality(self, path: Path) -> ModalityType:
        """Analyze text content to detect code vs documentation."""
        try:
            content = path.read_text(encoding='utf-8')[:2000]
            return self._analyze_content_type(content)
        except Exception:
            return ModalityType.TEXT
    
    def _analyze_content_type(self, content: str) -> ModalityType:
        """Analyze string content to determine type."""
        lines = content.split('\n')[:50]
        
        # Code detection heuristics
        code_patterns = [
            'def ', 'function ', 'class ', 'import ', 'from ',
            'const ', 'let ', 'var ', 'public ', 'private ',
            'async ', 'await ', '=>', '::', '->', '#include',
            'package ', 'namespace ', 'struct '
        ]
        
        code_score = sum(1 for line in lines for pattern in code_patterns if pattern in line)
        
        if code_score >= 3:
            return ModalityType.CODE
        
        # Table detection (CSV-like patterns)
        comma_lines = sum(1 for line in lines[:10] if ',' in line and not line.startswith('#'))
        if comma_lines >= 5 and content.count(',') > content.count('.') * 2:
            return ModalityType.TABLE
        
        return ModalityType.TEXT
    
    def detect_from_content(self, content: bytes, filename: str = "") -> ModalityType:
        """Detect modality from raw content bytes."""
        # Check magic bytes for images
        if content.startswith(b'\xff\xd8\xff'):  # JPEG
            return ModalityType.IMAGE
        elif content.startswith(b'\x89PNG'):  # PNG
            return ModalityType.IMAGE
        elif content.startswith(b'GIF8'):  # GIF
            return ModalityType.IMAGE
        elif content.startswith(b'RIFF') and content[8:12] == b'WEBP':  # WEBP
            return ModalityType.IMAGE
        
        # Try to decode as text
        try:
            text = content.decode('utf-8')
            return self._analyze_content_type(text)
        except:
            pass
        
        return ModalityType.TEXT
```

### Task 2.2: Create Test Data

```python
# Create test files for detection
import os

def create_test_data():
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Text file (documentation)
    (test_dir / "readme.txt").write_text("""
    System Documentation
    ====================
    
    This document describes the system architecture.
    The system consists of multiple microservices.
    Each service handles specific business logic.
    """)
    
    # Code file
    (test_dir / "example.py").write_text("""
    def calculate_sum(a, b):
        return a + b
    
    class Calculator:
        def __init__(self):
            self.result = 0
        
        def add(self, value):
            self.result += value
    """)
    
    # CSV file
    (test_dir / "data.csv").write_text("""
    Name,Age,City,Salary
    John,25,NYC,50000
    Jane,30,LA,60000
    Bob,35,Chicago,70000
    """)
    
    print("Test data created in test_data/ directory")

if __name__ == "__main__":
    create_test_data()
```

---

## Part 3: Embedding-Based Verification

### Task 3.1: Implement Embedding Detector

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModalityDetector:
    """
    Uses embedding similarity to verify content modality.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
        # Prototypical examples for each modality
        self.modality_prototypes = {
            ModalityType.TEXT: [
                "This document describes the system architecture.",
                "The following section explains the implementation details.",
                "According to the specifications, the component should integrate seamlessly."
            ],
            ModalityType.CODE: [
                "def calculate_sum(a, b): return a + b",
                "function fetchData() { return fetch('/api/data') }",
                "class UserService { constructor() { this.users = []; } }"
            ],
            ModalityType.TABLE: [
                "Name,Age,City\nJohn,25,NYC\nJane,30,LA",
                "| Product | Price | Quantity |\n| Item1 | $10 | 5 |",
                "id,value,timestamp\n1,100,2024-01-01\n2,200,2024-01-02"
            ],
            ModalityType.IMAGE: [
                "[Image: System architecture diagram showing components]",
                "[Figure 1: Data flow visualization with arrows]",
                "[Screenshot of the dashboard interface with charts]"
            ]
        }
        
        # Pre-compute prototype embeddings
        self.prototype_embeddings = {}
        for modality, texts in self.modality_prototypes.items():
            embeddings = self.model.encode(texts)
            self.prototype_embeddings[modality] = np.mean(embeddings, axis=0)
    
    def detect(self, content: str) -> ModalityPrediction:
        """Detect modality using embedding similarity."""
        # Encode the content
        content_embedding = self.model.encode([content])[0]
        
        # Calculate cosine similarity to each prototype
        similarities = {}
        for modality, prototype_emb in self.prototype_embeddings.items():
            # Cosine similarity
            dot_product = np.dot(content_embedding, prototype_emb)
            norm_content = np.linalg.norm(content_embedding)
            norm_proto = np.linalg.norm(prototype_emb)
            similarity = dot_product / (norm_content * norm_proto)
            similarities[modality] = similarity
        
        # Normalize to get confidence scores
        total = sum(similarities.values())
        confidences = {m: s/total for m, s in similarities.items()}
        
        # Determine primary and secondary
        sorted_mods = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_mods[0][0]
        primary_conf = sorted_mods[0][1]
        secondary = [m for m, c in sorted_mods[1:] if c > 0.15]
        
        return ModalityPrediction(
            primary=primary,
            confidence=round(primary_conf, 3),
            secondary=secondary,
            reasoning=f"Embedding similarity: {primary.value}={primary_conf:.3f}"
        )
```

---

## Part 4: Evaluation

### Task 4.1: Create Evaluation Dataset

```python
EVALUATION_DATA = [
    # (content, expected_modality)
    ("The system uses microservices architecture", ModalityType.TEXT),
    ("def hello(): print('world')", ModalityType.CODE),
    ("Name,Age\nJohn,25\nJane,30", ModalityType.TABLE),
    ("[Diagram: System architecture with 3 tiers]", ModalityType.IMAGE),
    ("This function calculates the sum of two numbers", ModalityType.TEXT),
    ("class User: def __init__(self, name): self.name = name", ModalityType.CODE),
    ("Product,Price,Stock\nWidget,10.99,100", ModalityType.TABLE),
    ("[Chart: Revenue growth over 12 months]", ModalityType.IMAGE),
]
```

### Task 4.2: Evaluate Detection Accuracy

```python
def evaluate_detector(detector, test_data):
    """Evaluate modality detector accuracy."""
    correct = 0
    total = len(test_data)
    
    for content, expected in test_data:
        prediction = detector.detect(content)
        is_correct = prediction.primary == expected
        
        if is_correct:
            correct += 1
        
        print(f"Content: {content[:50]}...")
        print(f"Predicted: {prediction.primary.value}, Expected: {expected.value}, Correct: {is_correct}")
        print()
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    # Test pattern-based detector
    pattern_detector = QueryModalityClassifier()
    print("=== Pattern-Based Detection ===\n")
    
    # Test embedding-based detector
    embedding_detector = EmbeddingModalityDetector()
    print("=== Embedding-Based Detection ===\n")
    evaluate_detector(embedding_detector, EVALUATION_DATA)
```

---

## 📝 Deliverables

1. Complete `solution.py` with all classes implemented
2. Test data files in `test_data/` directory
3. Evaluation results showing detection accuracy
4. Brief analysis of which method works better for different content types

## ✅ Success Criteria

- Pattern-based classifier achieves >80% accuracy on test queries
- Content detector correctly identifies file types
- Embedding-based verification provides confidence scores
- All tests pass without errors

## 🔍 Hints

- Use case-insensitive matching for pattern detection
- Normalize embeddings before calculating similarity
- Consider edge cases (empty content, mixed modalities)
- Test with real-world examples from your domain
