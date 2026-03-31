
import time
import sys
import os
import random
import string
import logging
import statistics

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipeline import RAGPipeline, RAGConfig
from src.retrieval import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_docs(n=100):
    """Generate n synthetic technical documents."""
    docs = []
    topics = ["Python", "Rust", "Machine Learning", "Kubernetes", "Docker", "React", "FastAPI", "Postgres"]
    
    logger.info(f"Generating {n} synthetic documents...")
    
    for i in range(n):
        topic = random.choice(topics)
        # Generate random "technical" content
        content_len = random.randint(50, 200)
        random_text = ''.join(random.choices(string.ascii_lowercase + " ", k=content_len))
        
        content = f"Topic: {topic}. Technical details: {random_text}. System ID: {random.randint(1000, 9999)}."
        
        docs.append(Document(
            id=f"doc_{i}",
            content=content,
            metadata={"topic": topic, "generated": True}
        ))
    return docs

def run_stress_test():
    print("="*60)
    print("🚀 WEEK 1 DAY 4: RAG STRESS TEST BENCHMARK")
    print("="*60)

    # 1. Setup
    config = RAGConfig(
        generator_model="gpt2", # Small model for benchmark speed
        dense_model="all-MiniLM-L6-v2",
        alpha=0.5,
        fusion="rrf",
        top_k=5,
        max_new_tokens=300
    )
    model = RAGPipeline(config)
    
    # 2. Indexing Benchmark
    docs = generate_synthetic_docs(n=100)

    start_time = time.time()
    model.index(docs)
    end_time = time.time()

    indexing_time = end_time - start_time
    print(f"\n[Indexing] 100 Documents")
    print(f"Total Time: {indexing_time:.4f}s")
    print(f"Avg Time/Doc: {indexing_time/100:.4f}s")

    # 3. Retrieval Latency Benchmark
    queries = [
        "How do I configure Docker?",
        "Python async await patterns",
        "Kubernetes pod lifecycle",
        "System ID 4059 error", # Specific ID lookup test
        "Machine learning model training"
    ]

    latencies = []
    print(f"\n[Retrieval] Running {len(queries)} queries...")

    for q in queries:
        t0 = time.time()
        # k=5 retrieval
        _ = model.retrieve(q, top_k=5)
        dt = time.time() - t0
        latencies.append(dt)
        print(f"  - Query: '{q[:20]}...' -> {dt:.4f}s")
        
    avg_lat = statistics.mean(latencies)
    max_lat = max(latencies)
    
    print(f"\n[Results]")
    print(f"Average Latency: {avg_lat:.4f}s")
    print(f"Max Latency:     {max_lat:.4f}s")
    print(f"Throughput:      {1/avg_lat:.2f} QPS (approx)")
    
    # 4. Pass/Fail Criteria
    # Arbitrary goals for a local laptop demo
    if avg_lat < 1.0: 
        print("\n✅ PASS: Average latency < 1.0s")
    else:
        print("\n⚠️ WARN: High latency detected")
        
    print("\nStress Test Complete.")

if __name__ == "__main__":
    run_stress_test()
