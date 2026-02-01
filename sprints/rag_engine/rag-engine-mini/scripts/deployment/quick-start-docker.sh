#!/bin/bash
# quick-start-docker.sh
# One-command deployment of RAG Engine using Docker Compose
# Usage: ./quick-start-docker.sh [environment]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
APP_NAME="rag-engine"
VERSION="latest"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     RAG Engine - Quick Docker Deployment              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"

# Check Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running${NC}"
    echo "Please start Docker service"
    exit 1
fi

echo -e "${GREEN}✅ Docker daemon is running${NC}"

# Create necessary directories
echo -e "${YELLOW}Step 2: Creating directories...${NC}"
mkdir -p data/postgres data/redis data/qdrant data/documents logs
echo -e "${GREEN}✅ Directories created${NC}"

# Generate secrets if they don't exist
echo -e "${YELLOW}Step 3: Setting up secrets...${NC}"
if [ ! -f .env ]; then
    echo "Generating environment variables..."
    
    JWT_SECRET=$(openssl rand -base64 32)
    DB_PASSWORD=$(openssl rand -base64 24)
    REDIS_PASSWORD=$(openssl rand -base64 24)
    API_KEY=$(openssl rand -base64 32)
    
    cat > .env <<EOF
# RAG Engine Environment Configuration
# Generated on $(date)
# Environment: ${ENVIRONMENT}

# Application
APP_NAME=${APP_NAME}
ENVIRONMENT=${ENVIRONMENT}
DEBUG=false
LOG_LEVEL=INFO
PORT=8000

# Security
JWT_SECRET=${JWT_SECRET}
API_KEY=${API_KEY}

# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=rag_engine
DB_USER=rag_user
DB_PASSWORD=${DB_PASSWORD}

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Storage
DOCUMENTS_PATH=/app/data/documents
MAX_DOCUMENT_SIZE=100MB

# Performance
WORKERS=4
MAX_REQUESTS=1000
TIMEOUT=300
EOF
    
    echo -e "${GREEN}✅ Environment file created (.env)${NC}"
    echo -e "${YELLOW}⚠️  Keep .env file secure - it contains secrets!${NC}"
else
    echo -e "${GREEN}✅ Environment file already exists${NC}"
fi

# Download docker-compose.yml if it doesn't exist
echo -e "${YELLOW}Step 4: Setting up Docker Compose...${NC}"
if [ ! -f docker-compose.yml ]; then
    echo "Downloading docker-compose.yml..."
    
    cat > docker-compose.yml <<'EOF'
version: '3.8'

services:
  api:
    image: rag-engine:latest
    container_name: rag-engine-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=${ENVIRONMENT}
      - DEBUG=${DEBUG:-false}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - JWT_SECRET=${JWT_SECRET}
      - API_KEY=${API_KEY}
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=${DB_NAME:-rag_engine}
      - DB_USER=${DB_USER:-rag_user}
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    volumes:
      - ./data/documents:/app/data/documents
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_started
    networks:
      - rag-engine-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:14-alpine
    container_name: rag-engine-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=${DB_NAME:-rag_engine}
      - POSTGRES_USER=${DB_USER:-rag_user}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - rag-engine-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-rag_user} -d ${DB_NAME:-rag_engine}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: rag-engine-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - ./data/redis:/data
    ports:
      - "6379:6379"
    networks:
      - rag-engine-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-engine-qdrant
    restart: unless-stopped
    volumes:
      - ./data/qdrant:/qdrant/storage
    ports:
      - "6333:6333"
    networks:
      - rag-engine-network

  nginx:
    image: nginx:alpine
    container_name: rag-engine-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    networks:
      - rag-engine-network

networks:
  rag-engine-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
EOF
    
    echo -e "${GREEN}✅ docker-compose.yml created${NC}"
else
    echo -e "${GREEN}✅ docker-compose.yml already exists${NC}"
fi

# Create nginx configuration
echo -e "${YELLOW}Step 5: Setting up Nginx...${NC}"
if [ ! -f nginx.conf ]; then
    cat > nginx.conf <<'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        location /health {
            proxy_pass http://api/health;
            access_log off;
        }
    }
}
EOF
    echo -e "${GREEN}✅ nginx.conf created${NC}"
fi

# Pull latest images
echo -e "${YELLOW}Step 6: Pulling Docker images...${NC}"
docker-compose pull
echo -e "${GREEN}✅ Images pulled${NC}"

# Start services
echo -e "${YELLOW}Step 7: Starting services...${NC}"
docker-compose up -d

# Wait for services to be healthy
echo -e "${YELLOW}Step 8: Waiting for services to be ready...${NC}"
sleep 10

# Check health
echo -e "${YELLOW}Step 9: Checking service health...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ All services are healthy!${NC}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Services failed to start properly${NC}"
    echo "Check logs with: docker-compose logs"
    exit 1
fi

# Display success message
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         🎉 Deployment Successful! 🎉                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Access your RAG Engine:${NC}"
echo -e "  API: ${GREEN}http://localhost:8000${NC}"
echo -e "  Health: ${GREEN}http://localhost:8000/health${NC}"
echo -e "  Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  View logs: ${YELLOW}docker-compose logs -f${NC}"
echo -e "  Stop: ${YELLOW}docker-compose down${NC}"
echo -e "  Restart: ${YELLOW}docker-compose restart${NC}"
echo -e "  Update: ${YELLOW}docker-compose pull && docker-compose up -d${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Environment file: ${YELLOW}.env${NC}"
echo -e "  Data directory: ${YELLOW}./data/${NC}"
echo -e "  Logs directory: ${YELLOW}./logs/${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Configure your LLM API keys in ${YELLOW}.env${NC}"
echo -e "  2. Upload documents via API or web interface"
echo -e "  3. Set up SSL certificates for production"
echo -e "  4. Configure monitoring and alerting"
echo ""
echo -e "${GREEN}Happy deploying! 🚀${NC}"
