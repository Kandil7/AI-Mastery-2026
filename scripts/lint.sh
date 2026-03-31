#!/bin/bash
# Lint Script for AI-Mastery-2026
# =================================
# Runs all code quality checks on the codebase.
#
# Usage:
#   ./scripts/lint.sh              # Lint all code
#   ./scripts/lint.sh --fix        # Auto-fix issues where possible
#   ./scripts/lint.sh src/         # Lint specific path

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
FIX_MODE=false
TARGET_PATH="src tests"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_MODE=true
            shift
            ;;
        *)
            TARGET_PATH="$1"
            shift
            ;;
    esac
done

echo "=================================="
echo "AI-Mastery-2026 Linting"
echo "=================================="
echo ""

# Function to run command with status
run_check() {
    local name=$1
    local cmd=$2
    
    echo -e "${YELLOW}Running ${name}...${NC}"
    if eval "$cmd"; then
        echo -e "${GREEN}✓ ${name} passed${NC}"
        return 0
    else
        echo -e "${RED}✗ ${name} failed${NC}"
        return 1
    fi
}

# Track failures
FAILURES=0

# Black (code formatting)
if [ "$FIX_MODE" = true ]; then
    run_check "Black (format)" "black $TARGET_PATH" || ((FAILURES++))
else
    run_check "Black (check)" "black --check $TARGET_PATH" || ((FAILURES++))
fi

# isort (import sorting)
if [ "$FIX_MODE" = true ]; then
    run_check "isort (sort)" "isort $TARGET_PATH" || ((FAILURES++))
else
    run_check "isort (check)" "isort --check $TARGET_PATH" || ((FAILURES++))
fi

# flake8 (linting)
run_check "flake8" "flake8 $TARGET_PATH" || ((FAILURES++))

# mypy (type checking)
run_check "mypy" "mypy $TARGET_PATH" || ((FAILURES++))

# Summary
echo ""
echo "=================================="
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILURES check(s) failed${NC}"
    if [ "$FIX_MODE" = false ]; then
        echo -e "${YELLOW}Tip: Run with --fix to auto-fix some issues${NC}"
    fi
    exit 1
fi
