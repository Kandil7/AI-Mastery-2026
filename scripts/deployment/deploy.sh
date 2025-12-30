#!/bin/bash
# =============================================================================
# Deployment Script for AI Engineer Toolkit
# =============================================================================
# Usage: ./deploy.sh [environment] [action]
# Environments: dev, staging, prod
# Actions: build, deploy, rollback, status

set -e

# Configuration
PROJECT_NAME="ai-mastery"
REGISTRY="${DOCKER_REGISTRY:-ghcr.io/ai-mastery-2026}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# BUILD
# =============================================================================

build() {
    log_info "Building Docker image..."
    
    docker build \
        --tag "${REGISTRY}/${PROJECT_NAME}:${VERSION}" \
        --tag "${REGISTRY}/${PROJECT_NAME}:latest" \
        --build-arg VERSION="${VERSION}" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        .
    
    log_info "Build complete: ${REGISTRY}/${PROJECT_NAME}:${VERSION}"
}

push() {
    log_info "Pushing to registry..."
    
    docker push "${REGISTRY}/${PROJECT_NAME}:${VERSION}"
    docker push "${REGISTRY}/${PROJECT_NAME}:latest"
    
    log_info "Push complete"
}

# =============================================================================
# DEPLOY
# =============================================================================

deploy_dev() {
    log_info "Deploying to development..."
    
    docker-compose -f docker-compose.yml up -d
    
    log_info "Development deployment complete"
    log_info "API: http://localhost:8000"
    log_info "Jupyter: http://localhost:8888"
}

deploy_staging() {
    log_info "Deploying to staging..."
    
    # Use staging compose file if exists
    if [ -f "docker-compose.staging.yml" ]; then
        docker-compose -f docker-compose.staging.yml up -d
    else
        ENVIRONMENT=staging docker-compose up -d
    fi
    
    log_info "Staging deployment complete"
}

deploy_prod() {
    log_info "Deploying to production..."
    
    # Production safety checks
    if [ -z "${CONFIRM_PROD}" ]; then
        log_error "Set CONFIRM_PROD=yes to deploy to production"
        exit 1
    fi
    
    # Blue-green deployment
    log_info "Starting blue-green deployment..."
    
    # Tag current as previous
    docker tag "${REGISTRY}/${PROJECT_NAME}:current" "${REGISTRY}/${PROJECT_NAME}:previous" 2>/dev/null || true
    
    # Tag new as current
    docker tag "${REGISTRY}/${PROJECT_NAME}:${VERSION}" "${REGISTRY}/${PROJECT_NAME}:current"
    
    # Deploy
    docker-compose -f docker-compose.prod.yml up -d --no-deps api
    
    # Health check
    log_info "Running health checks..."
    sleep 5
    
    if curl -sf http://localhost:8000/health > /dev/null; then
        log_info "Health check passed"
    else
        log_error "Health check failed, rolling back..."
        rollback
        exit 1
    fi
    
    log_info "Production deployment complete"
}

# =============================================================================
# ROLLBACK
# =============================================================================

rollback() {
    log_warn "Rolling back to previous version..."
    
    docker tag "${REGISTRY}/${PROJECT_NAME}:previous" "${REGISTRY}/${PROJECT_NAME}:current"
    docker-compose -f docker-compose.prod.yml up -d --no-deps api
    
    log_info "Rollback complete"
}

# =============================================================================
# STATUS
# =============================================================================

status() {
    log_info "Service Status:"
    docker-compose ps
    
    echo ""
    log_info "Container Health:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    log_info "Recent Logs:"
    docker-compose logs --tail=10
}

# =============================================================================
# MAIN
# =============================================================================

ENV="${1:-dev}"
ACTION="${2:-deploy}"

case "$ACTION" in
    build)
        build
        ;;
    push)
        build
        push
        ;;
    deploy)
        case "$ENV" in
            dev)
                deploy_dev
                ;;
            staging)
                deploy_staging
                ;;
            prod|production)
                deploy_prod
                ;;
            *)
                log_error "Unknown environment: $ENV"
                exit 1
                ;;
        esac
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 [env] [action]"
        echo "  Environments: dev, staging, prod"
        echo "  Actions: build, push, deploy, rollback, status"
        exit 1
        ;;
esac
