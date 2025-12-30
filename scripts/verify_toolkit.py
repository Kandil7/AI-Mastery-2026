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
import numpy as np

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_toolkit")

def verify_core():
    logger.info("Verifying Core Modules...")
    try:
        from src.core.math_operations import dot
        from src.core.optimization import GradientDescent
        from src.core.probability import Gaussian
        
        # Test Math
        v1, v2 = [1, 2], [3, 4]
        assert dot(v1, v2) == 11, "Dot product failed"
        
        # Test Prob
        g = Gaussian(mu=0, sigma=1)
        pdf = g.pdf(0)
        assert abs(pdf - 0.3989) < 0.001, "Gaussian PDF failed"
        
        logger.info("✅ Core Modules OK")
    except Exception as e:
        logger.error(f"❌ Core Modules Failed: {e}")
        return False
    return True

def verify_ml():
    logger.info("Verifying ML Modules...")
    try:
        from src.ml.classical import LinearRegression
        from src.ml.deep_learning import MLP
        
        # Test LR
        model = LinearRegression()
        assert model is not None
        
        # Test MLP
        mlp = MLP(input_size=10, hidden_sizes=[20], output_size=1)
        assert mlp is not None
        
        logger.info("✅ ML Modules OK")
    except Exception as e:
        logger.error(f"❌ ML Modules Failed: {e}")
        return False
    return True

def verify_llm():
    logger.info("Verifying LLM Modules...")
    try:
        from src.llm.attention import MultiHeadAttention
        from src.llm.rag import RAGPipeline
        from src.llm.fine_tuning import LoRALayer
        from src.llm.agents import ReActAgent
        
        # Test ReAct
        agent = ReActAgent(name="test", system_prompt="", llm_fn=lambda x: "Final Answer: OK")
        assert agent is not None
        
        # Test RAG
        rag = RAGPipeline()
        assert rag is not None
        
        logger.info("✅ LLM Modules OK")
    except Exception as e:
        logger.error(f"❌ LLM Modules Failed: {e}")
        return False
    return True

def verify_production():
    logger.info("Verifying Production Modules...")
    try:
        from src.production.api import create_app
        from src.production.caching import LRUCache, RedisCache
        from src.production.deployment import ModelRegistry
        
        # Test Cache
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        assert cache.get("a") == 1
        
        logger.info("✅ Production Modules OK")
    except Exception as e:
        logger.error(f"❌ Production Modules Failed: {e}")
        return False
    return True

def main():
    logger.info("Starting Toolkit Verification...")
    
    results = [
        verify_core(),
        verify_ml(),
        verify_llm(),
        verify_production()
    ]
    
    if all(results):
        logger.info("\n🎉 SUCCESS: AI Engineer Toolkit is completely installed and verified! 🎉")
        print("\nYou can now start exploring:")
        print("  - Notebooks: notebooks/")
        print("  - Case Studies: case_studies/")
        print("  - Run Tests: pytest tests/")
    else:
        logger.error("\n⚠️  WARNING: Some modules failed verification. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
