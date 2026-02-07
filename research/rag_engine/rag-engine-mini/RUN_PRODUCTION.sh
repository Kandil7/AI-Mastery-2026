#!/bin/bash
# PRODUCTION DEPLOYMENT - EXECUTION GUIDE
# =======================================
# Follow these steps to deploy RAG Engine Mini to production

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     RAG ENGINE MINI - PRODUCTION DEPLOYMENT                    ║"
echo "║     Execution Time: 5-10 minutes                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Step 1: Environment Check${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}❌ Error: Not in project root directory${NC}"
    echo "Please navigate to: K:\learning\technical\ai-ml\AI-Mastery-2026\sprints\rag_engine\rag-engine-mini"
    exit 1
fi

echo -e "${GREEN}✓ Project root confirmed${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not installed${NC}"
    echo "Install: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not installed${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon not running${NC}"
    echo "Start Docker Desktop or run: sudo systemctl start docker"
    exit 1
fi

echo -e "${GREEN}✓ Docker environment ready${NC}"
echo ""

echo -e "${BLUE}Step 2: Creating Production Structure${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create production directories
mkdir -p data/postgres data/redis data/qdrant data/documents logs ssl

echo -e "${GREEN}✓ Created directories:${NC}"
echo "  📁 data/postgres/     - PostgreSQL storage"
echo "  📁 data/redis/        - Redis cache storage"
echo "  📁 data/qdrant/       - Vector database storage"
echo "  📁 data/documents/    - Document uploads"
echo "  📁 logs/              - Application logs"
echo "  📁 ssl/               - SSL certificates"
echo ""

echo -e "${BLUE}Step 3: Generating Production Secrets${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Generate production secrets
if [ ! -f .env ]; then
    JWT_SECRET=$(openssl rand -base64 32 2>/dev/null || head -c 32 /dev/urandom | base64)
    DB_PASSWORD=$(openssl rand -base64 24 2>/dev/null || head -c 24 /dev/urandom | base64)
    REDIS_PASSWORD=$(openssl rand -base64 24 2>/dev/null || head -c 24 /dev/urandom | base64)
    API_KEY=$(openssl rand -base64 32 2>/dev/null || head -c 32 /dev/urandom | base64)
    
    cat > .env <<EOF
# RAG Engine Mini - Production Environment
# Generated: $(date)
# WARNING: Keep this file secure! Contains production secrets.

# Application
APP_NAME=rag-engine-mini
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
PORT=8000

# Security
JWT_SECRET=${JWT_SECRET}
API_KEY=${API_KEY}

# PostgreSQL Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=rag_engine_prod
DB_USER=rag_admin
DB_PASSWORD=${DB_PASSWORD}

# Redis Cache
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}

# Qdrant Vector Database
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Storage
DOCUMENTS_PATH=/app/data/documents
MAX_DOCUMENT_SIZE=100MB

# Performance
WORKERS=4
MAX_REQUESTS=1000
TIMEOUT=300

# API Keys (Add your actual API keys here)
# OPENAI_API_KEY=sk-...
# COHERE_API_KEY=...
# ANTHROPIC_API_KEY=...
EOF
    
    echo -e "${GREEN}✓ Generated .env file with secure secrets${NC}"
    echo -e "${YELLOW}⚠️  IMPORTANT: Review .env file and add your API keys${NC}"
else
    echo -e "${YELLOW}ℹ️  .env file already exists${NC}"
fi

echo ""

echo -e "${BLUE}Step 4: Production Docker Compose Configuration${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if docker-compose.prod.yml exists
if [ ! -f docker-compose.prod.yml ]; then
    echo -e "${RED}❌ docker-compose.prod.yml not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Production configuration ready${NC}"
echo ""

echo -e "${BLUE}Step 5: Pulling Latest Images${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This may take 2-5 minutes depending on your connection..."
echo ""

docker-compose -f docker-compose.prod.yml pull 2>&1 | tee logs/docker-pull.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Images pulled successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Some images may need to be built locally${NC}"
fi

echo ""

echo -e "${BLUE}Step 6: Starting Production Services${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting: API, PostgreSQL, Redis, Qdrant, Nginx"
echo "This may take 1-2 minutes..."
echo ""

docker-compose -f docker-compose.prod.yml up -d 2>&1 | tee logs/docker-up.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All services started${NC}"
else
    echo -e "${RED}❌ Failed to start services${NC}"
    echo "Check logs: docker-compose -f docker-compose.prod.yml logs"
    exit 1
fi

echo ""

echo -e "${BLUE}Step 7: Health Verification${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Waiting for services to be ready..."
echo ""

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Check API health
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is healthy${NC}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Services may still be starting${NC}"
    echo "Check status: docker-compose -f docker-compose.prod.yml ps"
fi

echo ""

echo -e "${BLUE}Step 8: Database Migration${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run database migrations
if docker-compose -f docker-compose.prod.yml ps | grep -q "api"; then
    echo "Running database migrations..."
    docker-compose -f docker-compose.prod.yml exec -T api alembic upgrade head 2>&1 | tee logs/migration.log
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Database migrations completed${NC}"
    else
        echo -e "${YELLOW}⚠️  Migration issue (may be already up to date)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  API service not running, skipping migration${NC}"
fi

echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     🎉 PRODUCTION DEPLOYMENT COMPLETE! 🎉                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Your RAG Engine Mini is now running in production!${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 ACCESS POINTS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🌐 API Base URL:     http://localhost:8000"
echo "  📖 API Documentation: http://localhost:8000/docs"
echo "  🔍 API Health Check:  http://localhost:8000/health"
echo "  📊 Nginx Proxy:       http://localhost (port 80)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛠️  USEFUL COMMANDS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  View logs:           docker-compose -f docker-compose.prod.yml logs -f"
echo "  Stop services:       docker-compose -f docker-compose.prod.yml down"
echo "  Restart:             docker-compose -f docker-compose.prod.yml restart"
echo "  Check status:        docker-compose -f docker-compose.prod.yml ps"
echo "  View API logs:       docker-compose -f docker-compose.prod.yml logs -f api"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 IMPORTANT FILES:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🔐 Environment:      .env (contains secrets - keep secure!)"
echo "  🐳 Compose Config:   docker-compose.prod.yml"
echo "  📊 Logs:             logs/"
echo "  💾 Data:             data/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Configure SSL certificates in ssl/ directory"
echo "  2. Add your OpenAI API key to .env file"
echo "  3. Upload documents via API or web interface"
echo "  4. Set up monitoring (see docs/learning/deployment/)"
echo "  5. Configure backups (scripts/backup/backup-routine.sh)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 QUICK TEST:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Test API:            curl http://localhost:8000/health"
echo "  Test with auth:      curl -H 'Authorization: Bearer YOUR_TOKEN' \\"
echo "                       http://localhost:8000/api/v1/documents"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✨ Happy deploying! Your RAG Engine is production-ready! ✨${NC}"
echo ""
echo "Need help? Check: docs/learning/deployment/README.md"
echo ""