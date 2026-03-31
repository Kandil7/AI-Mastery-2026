"""
Lab 1 Solution: Modality Detection and Classification

This module implements complete modality detection for adaptive multimodal RAG.
"""

from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ModalityType(Enum):
    """Supported content modalities."""
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    TABLE = "table"
    MIXED = "mixed"


@dataclass
class ModalityPrediction:
    """Result of modality classification."""
    primary: ModalityType
    confidence: float
    secondary: List[ModalityType]
    reasoning: str


# ============================================================================
# QUERY MODALITY CLASSIFIER
# ============================================================================

class QueryModalityClassifier:
    """
    Classifies queries by expected response modality.
    Uses pattern matching for fast, interpretable classification.
    """
    
    IMAGE_INDICATORS = [
        "diagram", "chart", "graph", "image", "picture", "photo",
        "screenshot", "visual", "illustration", "architecture",
        "flow", "map", "schema", "blueprint", "drawing", "figure",
        "plot", "visualization", "graphic", "rendering"
    ]
    
    CODE_INDICATORS = [
        "code", "snippet", "function", "method", "class",
        "implementation", "example", "script", "query",
        "api", "endpoint", "algorithm", "pattern", "def ",
        "import ", "function(", "class ", "=>", "async ",
        "await ", "const ", "let ", "var ", "public ", "private "
    ]
    
    TABLE_INDICATORS = [
        "table", "spreadsheet", "data", "metrics", "statistics",
        "comparison", "matrix", "grid", "specifications",
        "parameters", "configuration", "settings", "csv",
        "columns", "rows", "values", "list", "array",
        "json", "xml", "structured"
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
        
        # Check for mixed modality (multiple high scores)
        high_scores = [m for m, s in scores.items() if s >= 2]
        if len(high_scores) >= 2:
            return ModalityPrediction(
                primary=ModalityType.MIXED,
                confidence=0.8,
                secondary=high_scores,
                reasoning=f"Multiple modalities detected: {', '.join(m.value for m in high_scores)}"
            )
        
        # Determine primary modality
        primary = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[primary] / total if total > 0 else 0.25
        
        # Get secondary modalities
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
        ][:3]
        
        reasoning = f"Detected {scores[primary]} indicators for {primary.value}: {', '.join(found_indicators)}"
        
        return ModalityPrediction(
            primary=primary,
            confidence=round(confidence, 3),
            secondary=secondary,
            reasoning=reasoning
        )
    
    def classify_batch(self, queries: List[str]) -> List[ModalityPrediction]:
        """Classify multiple queries efficiently."""
        return [self.classify(query) for query in queries]


# ============================================================================
# CONTENT MODALITY DETECTOR
# ============================================================================

class ContentModalityDetector:
    """
    Detects modality of content for proper indexing.
    Supports file-based and content-based detection.
    """
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico', '.tiff'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', 
                       '.sql', '.sh', '.bash', '.yaml', '.yml', '.xml', '.html', '.css', '.scss'}
    TABLE_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.tsv', '.parquet', '.json', '.avro'}
    
    CODE_PATTERNS = [
        'def ', 'function ', 'class ', 'import ', 'from ',
        'const ', 'let ', 'var ', 'public ', 'private ',
        'async ', 'await ', '=>', '::', '->', '#include',
        'package ', 'namespace ', 'struct ', 'interface ',
        'enum ', 'typedef ', '#define ', '@Override', '@Entity'
    ]
    
    def detect_from_file(self, file_path: str) -> ModalityType:
        """Detect modality from file path and content."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        # Check extension first
        if ext in self.IMAGE_EXTENSIONS:
            return ModalityType.IMAGE
        elif ext in self.CODE_EXTENSIONS:
            return ModalityType.CODE
        elif ext in self.TABLE_EXTENSIONS:
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
        code_score = sum(
            1 for line in lines 
            for pattern in self.CODE_PATTERNS 
            if pattern in line
        )
        
        if code_score >= 3:
            return ModalityType.CODE
        
        # Table detection (CSV-like patterns)
        if len(lines) >= 3:
            comma_lines = sum(1 for line in lines[:10] if ',' in line and not line.startswith('#'))
            if comma_lines >= 5 and content.count(',') > content.count('.') * 2:
                return ModalityType.TABLE
        
        # JSON detection
        if content.strip().startswith('{') and content.strip().endswith('}'):
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
        elif content.startswith(b'RIFF') and len(content) > 12 and content[8:12] == b'WEBP':
            return ModalityType.IMAGE
        elif content.startswith(b'\x42\x4d'):  # BMP
            return ModalityType.IMAGE
        
        # Try to decode as text
        try:
            text = content.decode('utf-8')
            return self._analyze_content_type(text)
        except UnicodeDecodeError:
            pass
        
        return ModalityType.TEXT
    
    def detect_from_string(self, content: str) -> ModalityType:
        """Detect modality from string content."""
        return self._analyze_content_type(content)


# ============================================================================
# EMBEDDING-BASED MODALITY DETECTOR
# ============================================================================

class EmbeddingModalityDetector:
    """
    Uses embedding similarity to verify content modality.
    More accurate than pattern matching for ambiguous content.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._available = True
        except ImportError:
            self._available = False
            print("Warning: sentence-transformers not installed. Using fallback detector.")
        
        # Prototypical examples for each modality
        self.modality_prototypes = {
            ModalityType.TEXT: [
                "This document describes the system architecture and design patterns.",
                "The following section explains the implementation details and best practices.",
                "According to the specifications, the component should integrate seamlessly with existing systems.",
                "The documentation provides comprehensive guidance for developers and users."
            ],
            ModalityType.CODE: [
                "def calculate_sum(a, b): return a + b",
                "function fetchData() { return fetch('/api/data').then(r => r.json()) }",
                "class UserService { constructor() { this.users = []; } addUser(user) { this.users.push(user); } }",
                "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())"
            ],
            ModalityType.TABLE: [
                "Name,Age,City,Salary\nJohn,25,NYC,50000\nJane,30,LA,60000",
                "| Product | Price | Quantity | Total |\n| Item1 | $10 | 5 | $50 |",
                "id,value,timestamp,category\n1,100,2024-01-01,A\n2,200,2024-01-02,B",
                '{"name": "John", "age": 25, "city": "NYC"}'
            ],
            ModalityType.IMAGE: [
                "[Image: System architecture diagram showing three-tier architecture with database]",
                "[Figure 1: Data flow visualization with arrows indicating process sequence]",
                "[Screenshot of the dashboard interface with charts and metrics displayed]",
                "[Diagram: Component interaction showing API gateway and microservices]"
            ]
        }
        
        # Pre-compute prototype embeddings
        self.prototype_embeddings = {}
        if self._available:
            for modality, texts in self.modality_prototypes.items():
                embeddings = self.model.encode(texts)
                self.prototype_embeddings[modality] = np.mean(embeddings, axis=0)
    
    def detect(self, content: str) -> ModalityPrediction:
        """Detect modality using embedding similarity."""
        if not self._available:
            # Fallback to pattern-based detection
            detector = ContentModalityDetector()
            modality = detector.detect_from_string(content)
            return ModalityPrediction(
                primary=modality,
                confidence=0.5,
                secondary=[],
                reasoning="Fallback: sentence-transformers not available"
            )
        
        # Encode the content
        content_embedding = self.model.encode([content])[0]
        
        # Calculate cosine similarity to each prototype
        similarities = {}
        for modality, prototype_emb in self.prototype_embeddings.items():
            dot_product = np.dot(content_embedding, prototype_emb)
            norm_content = np.linalg.norm(content_embedding)
            norm_proto = np.linalg.norm(prototype_emb)
            
            if norm_content > 0 and norm_proto > 0:
                similarity = dot_product / (norm_content * norm_proto)
            else:
                similarity = 0
            similarities[modality] = similarity
        
        # Normalize to get confidence scores using softmax
        exp_sims = {m: np.exp(s * 3) for m, s in similarities.items()}  # Temperature scaling
        total = sum(exp_sims.values())
        confidences = {m: e/total for m, e in exp_sims.items()}
        
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


# ============================================================================
# HYBRID MODALITY DETECTOR
# ============================================================================

class HybridModalityDetector:
    """
    Combines pattern-based and embedding-based detection for robust classification.
    """
    
    def __init__(self, embedding_weight: float = 0.6):
        self.pattern_detector = ContentModalityDetector()
        self.embedding_detector = EmbeddingModalityDetector()
        self.embedding_weight = embedding_weight
    
    def detect(self, content: str) -> ModalityPrediction:
        """
        Detect modality using hybrid approach.
        
        Args:
            content: Content to analyze
            
        Returns:
            Combined modality prediction
        """
        # Get predictions from both methods
        pattern_pred = self._pattern_to_prediction(self.pattern_detector.detect_from_string(content))
        embedding_pred = self.embedding_detector.detect(content)
        
        # If embedding detector is confident, use it
        if embedding_pred.confidence > 0.7:
            return embedding_pred
        
        # If pattern detector found clear signals, use it
        if pattern_pred.confidence > 0.8:
            return pattern_pred
        
        # Combine predictions weighted by confidence
        return self._combine_predictions(pattern_pred, embedding_pred)
    
    def _pattern_to_prediction(self, modality: ModalityType) -> ModalityPrediction:
        """Convert simple modality to prediction format."""
        return ModalityPrediction(
            primary=modality,
            confidence=0.6,
            secondary=[],
            reasoning="Pattern-based detection"
        )
    
    def _combine_predictions(self, pred1: ModalityPrediction, pred2: ModalityPrediction) -> ModalityPrediction:
        """Combine two predictions using weighted confidence."""
        if pred1.primary == pred2.primary:
            # Agreement - boost confidence
            combined_conf = (pred1.confidence * (1 - self.embedding_weight) + 
                           pred2.confidence * self.embedding_weight)
            return ModalityPrediction(
                primary=pred1.primary,
                confidence=min(combined_conf + 0.1, 1.0),
                secondary=pred1.secondary + pred2.secondary,
                reasoning=f"Combined: {pred1.reasoning} + {pred2.reasoning}"
            )
        else:
            # Disagreement - use higher confidence
            if pred2.confidence > pred1.confidence:
                return pred2
            return pred1


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

EVALUATION_DATA = [
    ("The system uses microservices architecture with API gateway", ModalityType.TEXT),
    ("def calculate_sum(a, b):\n    return a + b", ModalityType.CODE),
    ("Name,Age,City\nJohn,25,NYC\nJane,30,LA", ModalityType.TABLE),
    ("[Diagram: System architecture with 3 tiers and load balancer]", ModalityType.IMAGE),
    ("This function calculates the sum of two numbers efficiently", ModalityType.TEXT),
    ("class User:\n    def __init__(self, name):\n        self.name = name", ModalityType.CODE),
    ("Product,Price,Stock\nWidget,10.99,100\nGadget,25.50,50", ModalityType.TABLE),
    ("[Chart: Revenue growth over 12 months showing upward trend]", ModalityType.IMAGE),
    ("import requests\nresponse = requests.get('https://api.example.com')", ModalityType.CODE),
    ("The quarterly report shows significant improvement in metrics", ModalityType.TEXT),
]


def evaluate_detector(detector, test_data: List[tuple]) -> Dict:
    """
    Evaluate modality detector accuracy.
    
    Args:
        detector: Detector instance with detect() method
        test_data: List of (content, expected_modality) tuples
        
    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = len(test_data)
    results = []
    
    for content, expected in test_data:
        prediction = detector.detect(content)
        is_correct = prediction.primary == expected
        
        if is_correct:
            correct += 1
        
        results.append({
            'content': content[:60] + '...' if len(content) > 60 else content,
            'predicted': prediction.primary.value,
            'expected': expected.value,
            'correct': is_correct,
            'confidence': prediction.confidence
        })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def print_evaluation_report(report: Dict):
    """Print formatted evaluation report."""
    print("=" * 70)
    print("MODALITY DETECTION EVALUATION REPORT")
    print("=" * 70)
    print(f"Accuracy: {report['accuracy']:.2%} ({report['correct']}/{report['total']})")
    print()
    
    for result in report['results']:
        status = "✓" if result['correct'] else "✗"
        print(f"{status} Content: {result['content']}")
        print(f"  Predicted: {result['predicted']} (conf: {result['confidence']:.3f})")
        print(f"  Expected: {result['expected']}")
        print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Lab 1: Modality Detection and Classification")
    print("=" * 70)
    print()
    
    # Test Query Classifier
    print("1. Testing Query Modality Classifier")
    print("-" * 50)
    query_classifier = QueryModalityClassifier()
    
    test_queries = [
        "Show me the architecture diagram",
        "Write a function to calculate sum",
        "Display the revenue data in a table",
        "Explain how the system works",
        "Show me the code and the flow chart",
    ]
    
    for query in test_queries:
        result = query_classifier.classify(query)
        print(f"Query: {query}")
        print(f"  → {result.primary.value} (confidence: {result.confidence})")
        print(f"  → Reasoning: {result.reasoning}")
        print()
    
    # Test Content Detector
    print("2. Testing Content Modality Detector")
    print("-" * 50)
    content_detector = ContentModalityDetector()
    
    test_contents = [
        ("def hello(): print('world')", "inline code"),
        ("Name,Age,City\nJohn,25,NYC", "inline table"),
        ("[Image: Architecture diagram]", "inline image description"),
        ("The system consists of multiple components.", "inline text"),
    ]
    
    for content, description in test_contents:
        modality = content_detector.detect_from_string(content)
        print(f"{description}: {content[:40]}... → {modality.value}")
    
    print()
    
    # Test Embedding Detector
    print("3. Testing Embedding-Based Detector")
    print("-" * 50)
    embedding_detector = EmbeddingModalityDetector()
    
    # Test Hybrid Detector
    print("4. Testing Hybrid Detector")
    print("-" * 50)
    hybrid_detector = HybridModalityDetector()
    
    # Run evaluation
    print("5. Running Evaluation")
    print("-" * 50)
    report = evaluate_detector(hybrid_detector, EVALUATION_DATA)
    print_evaluation_report(report)
    
    print()
    print("=" * 70)
    print("Lab 1 Complete!")
    print("=" * 70)
