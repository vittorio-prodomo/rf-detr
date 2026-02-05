#!/bin/bash

# GPU Test Wrapper Script
# Usage: .modal/pytest.sh [pytest-args] [test-path]
# Examples:
#   .modal/pytest.sh                           # Run all tests with -v
#   .modal/pytest.sh -v tests/                 # Run all tests verbosely
#   .modal/pytest.sh -v tests/datasets/        # Run specific directory  
#   .modal/pytest.sh "-v -k test_name" tests/  # Run specific test pattern

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}GPU Test Runner${NC}"
echo -e "${BLUE}================================${NC}"

# Parse arguments
if [ $# -eq 0 ]; then
    # No arguments - run all tests with -v
    TEST_PATH="tests/"
    PYTEST_ARGS="-v"
elif [ $# -eq 1 ]; then
    # One argument - could be pytest args or test path
    if [[ "$1" == -* ]]; then
        # Starts with dash - it's pytest args
        PYTEST_ARGS="$1"
        TEST_PATH="tests/"
    else
        # Otherwise it's a test path
        TEST_PATH="$1"
        PYTEST_ARGS="-v"
    fi
elif [ $# -eq 2 ]; then
    # Two arguments
    PYTEST_ARGS="$1"
    TEST_PATH="$2"
else
    echo -e "${RED}Error: Too many arguments${NC}"
    echo "Usage: $0 [pytest-args] [test-path]"
    exit 1
fi

# Check if modal is installed (prefer python -m modal for reliability)
if ! python -m modal --version &> /dev/null; then
    echo -e "${RED}❌ Modal CLI not found. Installing...${NC}"
    python -m pip install modal
fi

# Check for Modal credentials
if [ -z "$MODAL_TOKEN_ID" ] || [ -z "$MODAL_TOKEN_SECRET" ]; then
    echo -e "${YELLOW}⚠️  Modal credentials not set in environment variables.${NC}"
    echo "Checking for modal configuration..."
    
    # Check if modal.toml exists or if modal is configured
    if [ ! -f "modal.toml" ] && ! python -m modal profile current &> /dev/null; then
        echo -e "${RED}No modal configuration found.${NC}"
        echo "Please run: python -m modal token new"
        echo "Or set environment variables: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"
        exit 1
    fi
fi

echo -e "${GREEN}✓${NC} Modal CLI ready"
echo -e "${GREEN}✓${NC} Test path: $TEST_PATH"
echo -e "${GREEN}✓${NC} Pytest args: $PYTEST_ARGS"
echo

# Make script executable (for first run)
chmod +x "$0" 2>/dev/null || true

# Run the modal test runner
python -m modal run .modal/test_runner.py --test-path "$TEST_PATH" --pytest-args "$PYTEST_ARGS"

EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Tests passed!${NC}"
else
    echo -e "${RED}❌ Tests failed with exit code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
