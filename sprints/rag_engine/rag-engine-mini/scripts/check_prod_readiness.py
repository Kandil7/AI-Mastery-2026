import os
import sys

# Add project root to sys.path to resolve 'src' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from src.core.config import settings
from src.core.bootstrap import get_container

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProdCheck")

def check_env():
    logger.info("🔍 Checking environment variables...")
    required = ["DATABASE_URL", "REDIS_URL", "QDRANT_HOST"]
    if settings.llm_backend == "openai":
        required.append("openai_api_key")
        
    missing = [r for r in required if not getattr(settings, r.lower(), None)]
    if missing:
        logger.error(f"❌ Missing required variables: {missing}")
        return False
    logger.info("✅ Core environment variables present.")
    return True

def check_db():
    logger.info("🔍 Checking Database connectivity...")
    try:
        container = get_container()
        # Ping DB
        from sqlalchemy import text
        with container["db_engine"].connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connected.")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def check_redis():
    logger.info("🔍 Checking Redis connectivity...")
    try:
        import redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        logger.info("✅ Redis connected.")
        return True
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False

def check_qdrant():
    logger.info("🔍 Checking Qdrant connectivity...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant_host, port=6333)
        client.get_collections()
        logger.info("✅ Qdrant connected.")
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        return False

def main():
    logger.info("🚀 Starting Production Readiness Audit...")
    
    checks = [
        check_env(),
        check_db(),
        check_redis(),
        check_qdrant()
    ]
    
    if all(checks):
        logger.info("🏆 SYSTEM IS PRODUCTION READY!")
        sys.exit(0)
    else:
        logger.error("🛑 CRITICAL FAILURES DETECTED. DO NOT DEPLOY.")
        sys.exit(1)

if __name__ == "__main__":
    main()
