"""
RAGAS Evaluation Script
========================
Automated evaluation of the RAG pipeline using the RAGAS framework.

تحليل جودة RAG باستخدام إطار RAGAS
"""

import asyncio
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.core.bootstrap import get_container
from src.domain.entities import TenantId

async def run_evaluation():
    print("🚀 Starting RAGAS Evaluation...")
    
    container = get_container()
    use_case = container["ask_hybrid_use_case"]
    tenant_id = "eval_user"
    
    # Golden Set: Question, Contexts (Ground Truth), and expected output
    # In a real scenario, you'd load this from a JSON/CSV
    test_data = [
        {
            "question": "What is the main architecture of RAG Engine Mini?",
            "ground_truth": "The project follows a modular Clean Architecture with domain, application, adapters, and API layers. It uses the Ports and Adapters (Hexagonal) pattern.",
        },
        {
            "question": "How does the system handle multi-modal data?",
            "ground_truth": "The system uses PyMuPDF for table extraction and a vision-enhanced pipeline that generates text descriptions for images, which are then indexed in Qdrant.",
        }
    ]
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in test_data:
        print(f"Evaluating: {item['question']}")
        
        # 1. Execute Retrieval + Generation
        # We use retrieval_only to get contexts, then full execute for answer
        retrieved_chunks = use_case.execute_retrieval_only(
            tenant_id=TenantId(tenant_id),
            question=item["question"]
        )
        
        full_answer = use_case.execute(
            request=type('Request', (), {
                'tenant_id': tenant_id,
                'question': item['question'],
                'document_id': None,
                'k_vec': 5,
                'k_kw': 5,
                'fused_limit': 10,
                'rerank_top_n': 3,
                'expand_query': False
            })
        )
        
        questions.append(item["question"])
        answers.append(full_answer.text)
        contexts.append([c.text for c in retrieved_chunks])
        ground_truths.append(item["ground_truth"])
    
    # Create Dataset for RAGAS
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    # Run Evaluation
    # Note: RAGAS typically requires an LLM (like OpenAI) to grade
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    print("\n--- RAGAS Results ---")
    print(result)
    
    # Save results
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("\n✅ Results saved to evaluation_results.csv")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
