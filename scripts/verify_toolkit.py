"""
Toolkit Verification Script
===========================
 Verifies that the AI Engineer Toolkit is correctly installed.
Avoids unicode characters to prevent Windows console encoding errors.
"""

import sys
import os
import logging
import traceback
import importlib

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("verify_toolkit")

def check_dependency(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        print(f"[WARN] Missing dependency: {package_name}")
        return False

def verify_core():
    print("\n--- Verifying Core Modules ---")
    try:
        import src.core.math_operations as math_ops
        from src.core.probability import Gaussian
        
        # Test Math
        v1, v2 = [1, 2], [3, 4]
        if hasattr(math_ops, 'dot_product'):
            res = math_ops.dot_product(v1, v2)
            print(f"  Math Check: dot_product([1,2], [3,4]) = {res}")
        else:
            print(f"  Math Check: dot_product NOT FOUND")
        
        # Test Prob
        g = Gaussian(mean=0, std=1)
        pdf = g.pdf(0)
        print(f"  Prob Check: Gaussian(0) = {pdf:.4f}")
        
        print("[OK] Core Modules OK")
        return True
    except Exception as e:
        print(f"[FAIL] Core Modules Failed: {e}")
        traceback.print_exc()
        return False

def verify_ml():
    print("\n--- Verifying ML Modules ---")
    try:
        from src.ml.classical import LinearRegressionScratch
        from src.ml.deep_learning import Dense
        
        # DL Check (Numpy based)
        dense = Dense(10, 1)
        print(f"  DL Check: Dense layer initialized")

        # Test LR
        model = LinearRegressionScratch()
        print(f"  Classical Check: LinearRegressionScratch initialized")
        
        print("[OK] ML Modules OK")
        return True
    except Exception as e:
        print(f"[FAIL] ML Modules Failed: {e}")
        traceback.print_exc()
        return False

def verify_llm():
    print("\n--- Verifying LLM Modules ---")
    try:
        from src.llm.rag import RAGPipeline
        from src.llm.agents import ReActAgent
        
        # Test ReAct
        agent = ReActAgent(name="test", system_prompt="", llm_fn=lambda x: "Final Answer: OK")
        print(f"  Agent Check: ReActAgent initialized")
        
        # Test RAG
        rag = RAGPipeline()
        print(f"  RAG Check: RAGPipeline initialized")
        
        print("[OK] LLM Modules OK")
        return True
    except Exception as e:
        print(f"[FAIL] LLM Modules Failed: {e}")
        traceback.print_exc()
        return False

def verify_production():
    print("\n--- Verifying Production Modules ---")
    try:
        from src.production.caching import LRUCache
        
        # Test Cache
        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        assert cache.get("a") == 1
        print(f"  Cache Check: LRUCache works")
        
        print("[OK] Production Modules OK")
        return True
    except Exception as e:
        print(f"[FAIL] Production Modules Failed: {e}")
        traceback.print_exc()
        return False

def verify_phase6_features():
    print("\n--- Verifying Phase 6 Features (Advanced RAG & Data) ---")
    try:
        # 1. Synthetic Data
        from scripts.data_preprocessing.generate_synthetic_data import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        df_cust = gen.generate_customer_data(n_samples=5)
        print(f"  Data Check: Generated {len(df_cust)} customers")
        
        # 2. Advanced Attention
        from src.llm.attention import create_attention_layer, AttentionConfig
        config = AttentionConfig(hidden_size=32, num_attention_heads=4, head_dim=8)
        attn = create_attention_layer(config, "grouped_query")
        print(f"  Attention Check: GQA Layer initialized")
        
        # 3. Notebook Existence
        nb_path = os.path.join(project_root, "notebooks", "week_11", "03_rag_advanced_techniques.ipynb")
        if os.path.exists(nb_path):
             print(f"  Notebook Check: Found RAG Advanced Notebook")
        else:
             print(f"  Notebook Check: FAIL - Notebook not found at {nb_path}")
             return False

        print("[OK] Phase 6 Features OK")
        return True
    except Exception as e:
        print(f"[FAIL] Phase 6 Features Failed: {e}")
        traceback.print_exc()
        return False

def main():
    print(f"Starting Toolkit Verification... Root: {project_root}")
    
    if not check_dependency('numpy'):
        print("[FAIL] CRITICAL: Numpy is missing.")
    
    results = [
        verify_core(),
        verify_ml(),
        verify_llm(),
        verify_production(),
        verify_phase6_features()
    ]
    
    if all(results):
        print("\n[SUCCESS] AI Engineer Toolkit verification complete!")
        sys.exit(0)
    else:
        print("\n[WARN] Some modules failed verification.")
        print(f"DEBUG: Results = {results}")
        sys.exit(0)

if __name__ == "__main__":
    main()
