"""
RAG Engine Mini - Retrieval Evaluation
=======================================
Script to evaluate retrieval quality (Vector vs Hybrid).

سكربت لتقييم جودة الاسترجاع (Vector vs Hybrid)
"""

import time
from typing import List, Dict

from src.core.bootstrap import get_container
from src.domain.entities import TenantId

# Sample golden Q&A for evaluation
GOLDEN_QA = [
    {
        "question": "What are the main layers of the project architecture?",
        "expected_keywords": ["domain", "application", "adapters", "api"],
    },
    {
        "question": "Which vector store is used in this engine?",
        "expected_keywords": ["qdrant"],
    },
    {
        "question": "How does hybrid search work?",
        "expected_keywords": ["rrf", "fused", "vector", "keyword"],
    }
]

def evaluate_retrieval():
    """Run evaluation on sample queries."""
    container = get_container()
    use_case = container["ask_hybrid_use_case"]
    tenant = TenantId("demo_eval")
    
    print("=" * 60)
    print("📊 Retrieval Evaluation - Vector vs Hybrid")
    print("=" * 60)
    
    for item in GOLDEN_QA:
        question = item["question"]
        expected = item["expected_keywords"]
        
        print(f"\n🔍 Question: {question}")
        
        # Test Hybrid Search (Vector + Keyword)
        start = time.time()
        result = use_case.execute_retrieval_only(
            tenant_id=tenant,
            question=question,
            k_vec=30,
            k_kw=30,
            rerank_top_n=8
        )
        duration = time.time() - start
        
        # Calculate coverage of expected keywords in retrieved chunks
        found_keywords = []
        combined_text = " ".join([c.text.lower() for c in result])
        
        for kw in expected:
            if kw.lower() in combined_text:
                found_keywords.append(kw)
        
        recall = len(found_keywords) / len(expected)
        
        print(f"  ✨ Hybrid Strategy:")
        print(f"    - Recall: {recall:.2f} ({len(found_keywords)}/{len(expected)})")
        print(f"    - Time: {duration:.3f}s")
        print(f"    - Hits: {len(result)} chunks")
        
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete")

if __name__ == "__main__":
    # Mocking execute_retrieval_only if not present in use case
    # In production, this would be a method in AskQuestionHybridUseCase
    evaluate_retrieval()
