"""
Toolkit Verification Script
===========================
Verifies that the AI Engineer Toolkit is correctly installed and all modules can be imported.
Runs a minimal sanity check on core components.

Usage:
    python scripts/verify_toolkit.py
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
        logger.warning(f"⚠️  Missing dependency: {package_name}. Related modules may fail.")
        return False

def verify_core():
    print("\n--- Verifying Core Modules ---")
    try:
        import src.core.math_operations as math_ops
        from src.core.optimization import GradientDescent
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
        
        print("✅ Core Modules OK")
        return True
    except Exception as e:
        print(f"❌ Core Modules Failed: {e}")
        traceback.print_exc()
        return False

def verify_ml():
    print("\n--- Verifying ML Modules ---")
    try:
        from src.ml.classical import LinearRegression
        
        # Check Torch for Deep Learning
        if check_dependency('torch'):
            from src.ml.deep_learning import MLP
            mlp = MLP(input_size=10, hidden_sizes=[20], output_size=1)
            print(f"  DL Check: MLP initialized")
        else:
            print("  Skipping Deep Learning verify (torch missing)")

        # Test LR
        model = LinearRegression()
        print(f"  Classical Check: LinearRegression initialized")
        
        print("✅ ML Modules OK")
        return True
    except Exception as e:
        print(f"❌ ML Modules Failed: {e}")
        traceback.print_exc()
        return False

def verify_llm():
    print("\n--- Verifying LLM Modules ---")
    try:
        # Check Torch for Attention/Fine-tuning
        has_torch = check_dependency('torch')
        
        if has_torch:
            from src.llm.attention import MultiHeadAttention
            from src.llm.fine_tuning import LoRALayer
            print("  Attention & Fine-tuning imported")
        else:
            print("  Skipping Attention/FT verify (torch missing)")

        from src.llm.rag import RAGPipeline
        from src.llm.agents import ReActAgent
        
        # Test ReAct
        agent = ReActAgent(name="test", system_prompt="", llm_fn=lambda x: "Final Answer: OK")
        print(f"  Agent Check: ReActAgent initialized")
        
        # Test RAG
        rag = RAGPipeline()
        print(f"  RAG Check: RAGPipeline initialized")
        
        print("✅ LLM Modules OK")
        return True
    except Exception as e:
        print(f"❌ LLM Modules Failed: {e}")
        traceback.print_exc()
        return False

def verify_production():
    print("\n--- Verifying Production Modules ---")
    try:
        # Check FastAPI for API
        if check_dependency('fastapi'):
            import src.production.api as api_module
            print("  API module imported")
        else:
            print("  Skipping API verify (fastapi missing)")
            
        from src.production.caching import LRUCache, RedisCache
        from src.production.deployment import ModelRegistry
        
        # Test Cache
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        assert cache.get("a") == 1
        print(f"  Cache Check: LRUCache works")
        
        print("✅ Production Modules OK")
        return True
    except Exception as e:
        print(f"❌ Production Modules Failed: {e}")
        traceback.print_exc()
        return False

def main():
    print(f"Starting Toolkit Verification... Root: {project_root}")
    
    # Check Numpy (Critical)
    if not check_dependency('numpy'):
        print("❌ CRITICAL: Numpy is missing. Detailed checks will likely fail.")
    
    results = [
        verify_core(),
        verify_ml(),
        verify_llm(),
        verify_production()
    ]
    
    if all(results):
        print("\n🎉 SUCCESS: AI Engineer Toolkit verification complete! 🎉")
    else:
        print("\n⚠️  WARNING: Some modules failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    main()
