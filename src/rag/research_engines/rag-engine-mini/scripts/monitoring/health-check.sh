#!/bin/bash
# health-check.sh
# Comprehensive health check for RAG Engine deployment
# Usage: ./health-check.sh [full|quick]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CHECK_TYPE=${1:-full}
API_URL=${API_URL:-http://localhost:8000}
TIMEOUT=10

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        RAG Engine Health Check                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}API URL: ${API_URL}${NC}"
echo -e "${BLUE}Type: ${CHECK_TYPE}${NC}"
echo ""

# Track results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    WARNINGS=$((WARNINGS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

# Check 1: API Health Endpoint
check_api_health() {
    echo -e "${YELLOW}Checking API health...${NC}"
    
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time ${TIMEOUT} ${API_URL}/health 2>/dev/null || echo "000")
    
    if [ "${response}" == "200" ]; then
        check_pass "API health endpoint (HTTP 200)"
        
        # Check response time
        response_time=$(curl -s -o /dev/null -w "%{time_total}" --max-time ${TIMEOUT} ${API_URL}/health 2>/dev/null || echo "999")
        if (( $(echo "${response_time} < 1.0" | bc -l) )); then
            check_pass "API response time (${response_time}s)"
        else
            check_warn "API response time slow (${response_time}s)"
        fi
    else
        check_fail "API health endpoint (HTTP ${response})"
    fi
}

# Check 2: Database Connectivity
check_database() {
    echo -e "${YELLOW}Checking database connectivity...${NC}"
    
    if curl -s --max-time ${TIMEOUT} ${API_URL}/health 2>/dev/null | grep -q "database"; then
        check_pass "Database connection"
    else
        check_fail "Database connection"
    fi
}

# Check 3: Redis/Cache
check_cache() {
    echo -e "${YELLOW}Checking cache service...${NC}"
    
    # Try to set and get a test value
    test_key="health_check_$(date +%s)"
    
    # This would need Redis CLI or API endpoint
    # For now, just check if Redis is responding
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping 2>/dev/null | grep -q "PONG"; then
            check_pass "Redis cache responsive"
        else
            check_fail "Redis cache not responding"
        fi
    else
        check_warn "Redis CLI not available, skipping cache check"
    fi
}

# Check 4: Qdrant Vector Store
check_qdrant() {
    echo -e "${YELLOW}Checking Qdrant vector store...${NC}"
    
    qdrant_response=$(curl -s -o /dev/null -w "%{http_code}" --max-time ${TIMEOUT} http://localhost:6333/collections 2>/dev/null || echo "000")
    
    if [ "${qdrant_response}" == "200" ]; then
        check_pass "Qdrant vector store"
    else
        check_warn "Qdrant check skipped (may not be exposed)"
    fi
}

# Check 5: Disk Space
check_disk_space() {
    echo -e "${YELLOW}Checking disk space...${NC}"
    
    # Check data directory
    if [ -d "./data" ]; then
        usage=$(df -h ./data | tail -1 | awk '{print $5}' | sed 's/%//')
        if [ ${usage} -lt 80 ]; then
            check_pass "Data directory disk usage (${usage}%)"
        elif [ ${usage} -lt 90 ]; then
            check_warn "Data directory disk usage high (${usage}%)"
        else
            check_fail "Data directory disk usage critical (${usage}%)"
        fi
    fi
    
    # Check logs directory
    if [ -d "./logs" ]; then
        log_size=$(du -sm ./logs 2>/dev/null | cut -f1)
        if [ ${log_size} -lt 1000 ]; then
            check_pass "Log directory size (${log_size}MB)"
        else
            check_warn "Log directory large (${log_size}MB)"
        fi
    fi
}

# Check 6: Memory Usage
check_memory() {
    echo -e "${YELLOW}Checking memory usage...${NC}"
    
    if command -v free &> /dev/null; then
        memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
        if [ ${memory_usage} -lt 80 ]; then
            check_pass "System memory usage (${memory_usage}%)"
        elif [ ${memory_usage} -lt 90 ]; then
            check_warn "System memory usage high (${memory_usage}%)"
        else
            check_fail "System memory usage critical (${memory_usage}%)"
        fi
    else
        check_warn "Cannot check memory (free command not available)"
    fi
}

# Check 7: CPU Usage
check_cpu() {
    echo -e "${YELLOW}Checking CPU usage...${NC}"
    
    if command -v top &> /dev/null; then
        # Get 1-second average
        cpu_idle=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print int($1)}')
        cpu_usage=$((100 - cpu_idle))
        
        if [ ${cpu_usage} -lt 70 ]; then
            check_pass "CPU usage (${cpu_usage}%)"
        elif [ ${cpu_usage} -lt 85 ]; then
            check_warn "CPU usage high (${cpu_usage}%)"
        else
            check_fail "CPU usage critical (${cpu_usage}%)"
        fi
    else
        check_warn "Cannot check CPU (top command not available)"
    fi
}

# Check 8: Container Status (if Docker)
check_containers() {
    echo -e "${YELLOW}Checking Docker containers...${NC}"
    
    if command -v docker-compose &> /dev/null; then
        running=$(docker-compose ps -q 2>/dev/null | wc -l)
        total=$(docker-compose config --services 2>/dev/null | wc -l)
        
        if [ ${running} -eq ${total} ]; then
            check_pass "All containers running (${running}/${total})"
        else
            check_fail "Some containers not running (${running}/${total})"
        fi
        
        # Check container health
        unhealthy=$(docker-compose ps 2>/dev/null | grep -c "unhealthy" || echo 0)
        if [ ${unhealthy} -eq 0 ]; then
            check_pass "All containers healthy"
        else
            check_warn "${unhealthy} container(s) unhealthy"
        fi
    else
        check_warn "Docker Compose not available"
    fi
}

# Check 9: SSL Certificate (if HTTPS)
check_ssl() {
    echo -e "${YELLOW}Checking SSL certificate...${NC}"
    
    if [[ ${API_URL} == https* ]]; then
        domain=$(echo ${API_URL} | sed 's|https://||' | cut -d'/' -f1 | cut -d':' -f1)
        
        expiry=$(echo | openssl s_client -servername ${domain} -connect ${domain}:443 2>/dev/null | \
            openssl x509 -noout -dates 2>/dev/null | grep notAfter | cut -d= -f2)
        
        if [ ! -z "${expiry}" ]; then
            expiry_epoch=$(date -d "${expiry}" +%s 2>/dev/null || echo 0)
            current_epoch=$(date +%s)
            days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
            
            if [ ${days_until_expiry} -gt 30 ]; then
                check_pass "SSL certificate valid (${days_until_expiry} days)"
            elif [ ${days_until_expiry} -gt 7 ]; then
                check_warn "SSL certificate expires in ${days_until_expiry} days"
            else
                check_fail "SSL certificate expires in ${days_until_expiry} days!"
            fi
        else
            check_warn "Could not check SSL certificate"
        fi
    else
        check_warn "SSL not enabled (using HTTP)"
    fi
}

# Check 10: API Endpoints
check_api_endpoints() {
    echo -e "${YELLOW}Checking API endpoints...${NC}"
    
    endpoints=(
        "/health"
        "/docs"
        "/api/v1/search"
    )
    
    for endpoint in "${endpoints[@]}"; do
        response=$(curl -s -o /dev/null -w "%{http_code}" --max-time ${TIMEOUT} ${API_URL}${endpoint} 2>/dev/null || echo "000")
        
        # Accept 200, 401 (auth required), 405 (method not allowed)
        if [[ "${response}" =~ ^(200|401|405)$ ]]; then
            check_pass "Endpoint ${endpoint} (HTTP ${response})"
        else
            check_fail "Endpoint ${endpoint} (HTTP ${response})"
        fi
    done
}

# Check 11: Log Errors
check_logs() {
    echo -e "${YELLOW}Checking recent errors in logs...${NC}"
    
    if [ -d "./logs" ]; then
        # Count errors in last hour
        error_count=$(find ./logs -name "*.log" -mtime -0.04 -exec grep -i "error" {} + 2>/dev/null | wc -l)
        
        if [ ${error_count} -eq 0 ]; then
            check_pass "No recent errors in logs"
        elif [ ${error_count} -lt 10 ]; then
            check_warn "${error_count} errors found in recent logs"
        else
            check_fail "${error_count} errors found in recent logs"
        fi
    else
        check_warn "Log directory not found"
    fi
}

# Check 12: Backup Status
check_backups() {
    echo -e "${YELLOW}Checking backup status...${NC}"
    
    if [ -d "/backups" ] || [ -d "./backups" ]; then
        backup_dir="/backups"
        if [ ! -d "${backup_dir}" ]; then
            backup_dir="./backups"
        fi
        
        # Find most recent backup
        latest_backup=$(find ${backup_dir} -maxdepth 1 -type d -name "20*" | sort | tail -1)
        
        if [ ! -z "${latest_backup}" ]; then
            backup_time=$(stat -c %Y ${latest_backup} 2>/dev/null || echo 0)
            current_time=$(date +%s)
            hours_since_backup=$(( (current_time - backup_time) / 3600 ))
            
            if [ ${hours_since_backup} -lt 25 ]; then
                check_pass "Recent backup available (${hours_since_backup}h ago)"
            elif [ ${hours_since_backup} -lt 48 ]; then
                check_warn "Last backup ${hours_since_backup}h ago"
            else
                check_fail "No recent backup (${hours_since_backup}h ago)"
            fi
        else
            check_fail "No backups found"
        fi
    else
        check_warn "Backup directory not configured"
    fi
}

# Quick health check
quick_check() {
    check_api_health
    check_database
    check_containers
}

# Full health check
full_check() {
    quick_check
    check_cache
    check_qdrant
    check_disk_space
    check_memory
    check_cpu
    check_ssl
    check_api_endpoints
    check_logs
    check_backups
}

# Main execution
main() {
    echo -e "${BLUE}Running checks...${NC}"
    echo ""
    
    case ${CHECK_TYPE} in
        quick)
            quick_check
            ;;
        full)
            full_check
            ;;
        *)
            echo -e "${RED}Unknown check type: ${CHECK_TYPE}${NC}"
            echo "Usage: $0 [quick|full]"
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}          HEALTH CHECK SUMMARY         ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "Total Checks: ${TOTAL_CHECKS}"
    echo -e "${GREEN}Passed: ${PASSED_CHECKS}${NC}"
    echo -e "${YELLOW}Warnings: ${WARNINGS}${NC}"
    echo -e "${RED}Failed: ${FAILED_CHECKS}${NC}"
    echo ""
    
    # Calculate health score
    if [ ${TOTAL_CHECKS} -gt 0 ]; then
        health_score=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
        echo -e "Health Score: ${health_score}%"
        echo ""
        
        if [ ${FAILED_CHECKS} -eq 0 ] && [ ${WARNINGS} -eq 0 ]; then
            echo -e "${GREEN}🎉 All systems operational!${NC}"
            exit 0
        elif [ ${FAILED_CHECKS} -eq 0 ]; then
            echo -e "${YELLOW}⚠️  Systems operational with warnings${NC}"
            exit 0
        else
            echo -e "${RED}❌ Some systems are experiencing issues${NC}"
            exit 1
        fi
    fi
}

# Run checks
main "$@"
