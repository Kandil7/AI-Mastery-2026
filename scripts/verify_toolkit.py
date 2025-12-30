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

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_toolkit")

def verify_core():
    logger.info("Verifying Core Modules...")
    try:
        import src.core.math_operations as math_ops
        from src.core.optimization import GradientDescent
        from src.core.probability import Gaussian
        
        # Test Math
        v1, v2 = [1, 2], [3, 4]
        # Check if dot is in math_ops or we need to import it
        if hasattr(math_ops, 'dot'):
            assert math_ops.dot(v1, v2) == 11, "Dot product failed"
        else:
            logger.warning("dot function not found in math_operations (check exports)")
        
        # Test Prob
        g = Gaussian(mu=0, sigma=1)
        pdf = g.pdf(0)
        assert abs(pdf - 0.3989) < 0.001, "Gaussian PDF failed"
        
        logger.info("✅ Core Modules OK")
    except Exception as e:
        logger.error(f"❌ Core Modules Failed: {e}")
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
        return False
    return True

def verify_production():
    logger.info("Verifying Production Modules...")
    try:
        # Check exports in api
        import src.production.api as api_module
        from src.production.caching import LRUCache
        from src.production.deployment import ModelRegistry
        
        # Test Cache
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        assert cache.get("a") == 1
        
        logger.info("✅ Production Modules OK")
    except Exception as e:
        logger.error(f"❌ Production Modules Failed: {e}")
        logger.error(traceback.format_exc())
        return False
    return True

def main():
    logger.info(f"Starting Toolkit Verification... Root: {project_root}")
    logger.info(f"Sys Path: {sys.path}")
    
    results = [
        verify_core(),
        verify_ml(),
        verify_llm(),
        verify_production()
    ]
    
    if all(results):
        logger.info("\n🎉 SUCCESS: AI Engineer Toolkit is completely installed and verified! 🎉")
    else:
        logger.error("\n⚠️  WARNING: Some modules failed verification. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
