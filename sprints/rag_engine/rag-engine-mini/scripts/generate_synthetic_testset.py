import json
import argparse
from typing import List
from src.core.bootstrap import get_container
from src.application.services.chunking import chunk_text_token_aware
from src.domain.entities import ChunkSpec

def generate_synthetic_testset(file_path: str, output_path: str, num_questions_per_chunk: int = 1):
    """
    Generate a synthetic dataset for RAGAS evaluation.
    """
    container = get_container()
    llm = container["llm"]
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 1. Chunk the document
    spec = ChunkSpec(max_tokens=600, overlap_tokens=50)
    chunks = chunk_text_token_aware(text, spec)
    
    testset = []
    
    print(f"Generating synthetic questions for {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        prompt = f"""
        Given the context below, generate {num_questions_per_chunk} question and answer pairs.
        The questions should be verifiable using ONLY the context.
        Format: JSON list of objects with "question" and "ground_truth".
        
        CONTEXT:
        {chunk}
        
        JSON:
        """
        try:
            response = llm.generate(prompt)
            # Find the JSON part
            start = response.find("[")
            end = response.rfind("]") + 1
            pairs = json.loads(response[start:end])
            
            for p in pairs:
                testset.append({
                    "question": p["question"],
                    "context": [chunk],
                    "ground_truth": p["ground_truth"]
                })
            print(f"Done chunk {i+1}/{len(chunks)}")
        except Exception as e:
            print(f"Error in chunk {i+1}: {e}")
            
    # 2. Save to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in testset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully generated {len(testset)} test cases to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="synthetic_testset.jsonl")
    args = parser.parse_args()
    
    generate_synthetic_testset(args.input, args.output)
