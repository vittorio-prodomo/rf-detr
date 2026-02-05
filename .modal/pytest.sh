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
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}GPU Test Runner${NC}"
echo -e "${BLUE}================================${NC}"

# Default values
TEST_PATH="${2:-tests/}"
PYTEST_ARGS="${1:--v}"

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    python -m pip install modal
fi

# Check for Modal credentials
if [ -z "$MODAL_TOKEN_ID" ] || [ -z "$MODAL_TOKEN_SECRET" ]; then
    echo "⚠️  Modal credentials not set."
    echo "Please run: modal token new"
    echo "Or set environment variables: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"
    exit 1
fi

echo -e "${GREEN}✓${NC} Modal CLI ready"
echo -e "${GREEN}✓${NC} Test path: $TEST_PATH"
echo -e "${GREEN}✓${NC} Pytest args: $PYTEST_ARGS"
echo

# Run the modal test runner
# Profile is configured in modal.toml
python -m modal run .modal/test_runner.py --test-path "$TEST_PATH" --pytest-args "$PYTEST_ARGS"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Tests passed!${NC}"
else
    echo -e "❌ Tests failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
