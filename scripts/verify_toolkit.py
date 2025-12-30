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
        if hasattr(math_ops, 'dot'):
            res = math_ops.dot(v1, v2)
            print(f"  Math Check: dot([1,2], [3,4]) = {res}")
        
        # Test Prob
        g = Gaussian(mu=0, sigma=1)
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
        from src.ml.classical import LinearRegressionScratch  # Updated class name
        
        # Check Torch
        if check_dependency('torch'):
            from src.ml.deep_learning import Dense
            dense = Dense(10, 1)
            print(f"  DL Check: Dense layer initialized")
        else:
            print("  Skipping Deep Learning verify (torch missing)")

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

def main():
    print(f"Starting Toolkit Verification... Root: {project_root}")
    
    if not check_dependency('numpy'):
        print("[FAIL] CRITICAL: Numpy is missing.")
    
    results = [
        verify_core(),
        verify_ml(),
        verify_llm(),
        verify_production()
    ]
    
    print(f"DEBUG: Results = {results}")
    
    if all(results):
        print("\n[SUCCESS] AI Engineer Toolkit verification complete!")
    else:
        print("\n[WARN] Some modules failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    main()
