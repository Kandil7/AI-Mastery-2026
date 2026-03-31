#!/bin/bash
# Test Script for AI-Mastery-2026
# =================================
# Runs all tests on the codebase.
#
# Usage:
#   ./scripts/test.sh                # Run all tests
#   ./scripts/test.sh --unit         # Run only unit tests
#   ./scripts/test.sh --integration  # Run only integration tests
#   ./scripts/test.sh --coverage     # Run with coverage report
#   ./scripts/test.sh tests/unit/core  # Run specific test path

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
TARGET_PATH="tests"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            TARGET_PATH="tests/unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            TARGET_PATH="tests/integration"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            TARGET_PATH="$1"
            shift
            ;;
    esac
done

echo "=================================="
echo "AI-Mastery-2026 Testing"
echo "=================================="
echo -e "${BLUE}Test Type:${NC} $TEST_TYPE"
echo -e "${BLUE}Target:${NC} $TARGET_PATH"
echo -e "${BLUE}Coverage:${NC} $COVERAGE"
echo ""

# Build pytest command
PYTEST_CMD="pytest"

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term-missing"
fi

# Add test markers based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD -m 'unit or not integration'"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD -m integration"
        ;;
esac

# Add target path
PYTEST_CMD="$PYTEST_CMD $TARGET_PATH"

# Run tests
echo "=================================="
echo -e "${YELLOW}Running tests...${NC}"
echo ""

if eval "$PYTEST_CMD"; then
    echo ""
    echo "=================================="
    echo -e "${GREEN}All tests passed!${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${BLUE}Coverage report generated:${NC}"
        echo "  - HTML: htmlcov/index.html"
    fi
    
    exit 0
else
    echo ""
    echo "=================================="
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
