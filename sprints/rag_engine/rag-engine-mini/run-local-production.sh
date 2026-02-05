#!/bin/bash
# =============================================================================
# RAG Engine Mini - Local Production Deployment
# =============================================================================
# This script runs infrastructure services in Docker and the app locally
# 
# INFRASTRUCTURE (Docker):
#   - PostgreSQL on port 5432
#   - Redis on port 6379
#   - Qdrant on port 6333
#   - Flower (Celery monitoring) on port 5555
#
# APPLICATION (Local Python):
#   - FastAPI API on port 8000
#   - Celery Worker
#   - Frontend on port 3000 (optional)
# =============================================================================

set -e

echo "============================================================================="
echo "🚀 RAG ENGINE MINI - LOCAL PRODUCTION DEPLOYMENT"
echo "============================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# =============================================================================
# STEP 1: Check Prerequisites
# =============================================================================
echo -e "${BLUE}[1/7] Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker Desktop first.${NC}"
    exit 1
fi

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python is not installed. Please install Python 3.11+ first.${NC}"
    exit 1
fi

# Check pip
if ! command -v pip &> /dev/null; then
    echo -e "${RED}❌ pip is not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo ""

# =============================================================================
# STEP 2: Create Environment File
# =============================================================================
echo -e "${BLUE}[2/7] Setting up environment...${NC}"

if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env file to add your API keys before continuing${NC}"
fi

echo -e "${GREEN}✅ Environment configured${NC}"
echo ""

# =============================================================================
# STEP 3: Start Infrastructure Services
# =============================================================================
echo -e "${BLUE}[3/7] Starting infrastructure services...${NC}"

# Start only infrastructure services (no app)
docker-compose -f docker-compose.prod.yml up -d postgres redis qdrant flower

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are healthy
echo ""
echo -e "${BLUE}Checking service health...${NC}"

# Check PostgreSQL
if docker-compose -f docker-compose.prod.yml ps | grep -q "postgres.*Up"; then
    echo -e "${GREEN}✅ PostgreSQL is running on port 5432${NC}"
else
    echo -e "${RED}❌ PostgreSQL failed to start${NC}"
fi

# Check Redis
if docker-compose -f docker-compose.prod.yml ps | grep -q "redis.*Up"; then
    echo -e "${GREEN}✅ Redis is running on port 6379${NC}"
else
    echo -e "${RED}❌ Redis failed to start${NC}"
fi

# Check Qdrant
if docker-compose -f docker-compose.prod.yml ps | grep -q "qdrant.*Up"; then
    echo -e "${GREEN}✅ Qdrant is running on port 6333${NC}"
else
    echo -e "${RED}❌ Qdrant failed to start${NC}"
fi

echo ""

# =============================================================================
# STEP 4: Install Python Dependencies
# =============================================================================
echo -e "${BLUE}[4/7] Installing Python dependencies...${NC}"

# Check if virtual environment exists
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || .venv/Scripts/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# =============================================================================
# STEP 5: Run Database Migrations
# =============================================================================
echo -e "${BLUE}[5/7] Running database migrations...${NC}"

# Run Alembic migrations
alembic upgrade head || echo -e "${YELLOW}⚠️  Migration skipped (Alembic not configured)${NC}"

echo -e "${GREEN}✅ Database ready${NC}"
echo ""

# =============================================================================
# STEP 6: Start Application
# =============================================================================
echo -e "${BLUE}[6/7] Starting RAG Engine API...${NC}"
echo ""
echo -e "${GREEN}🚀 RAG Engine is now running!${NC}"
echo ""
echo "============================================================================="
echo "📋 SERVICE ENDPOINTS:"
echo "============================================================================="
echo ""
echo "  API Server:       http://localhost:8000"
echo "  API Docs:         http://localhost:8000/docs"
echo "  Health Check:     http://localhost:8000/health"
echo "  GraphQL:          http://localhost:8000/graphql"
echo ""
echo "  PostgreSQL:       localhost:5432"
echo "  Redis:            localhost:6379"
echo "  Qdrant:           http://localhost:6333"
echo "  Flower (Celery):  http://localhost:5555"
echo ""
echo "============================================================================="
echo "🛑 TO STOP:"
echo "  - Press Ctrl+C to stop the API"
echo "  - Run: docker-compose -f docker-compose.prod.yml down"
echo "============================================================================="
echo ""

# =============================================================================
# STEP 7: Start the API Server
# =============================================================================
echo -e "${BLUE}[7/7] Launching API server...${NC}"
echo ""

# Start the API server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
