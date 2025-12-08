#!/bin/bash
# =============================================================================
# Evaluation Script for GPT-OSS MixLoRA
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default values
TASKS=${TASKS:-"arc_easy"}
LIMIT=${LIMIT:-""}
ADAPTER_REPO=${ADAPTER_REPO:-""}
DEVICE=${DEVICE:-"cuda:0"}
DTYPE=${DTYPE:-"bfloat16"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --adapter_repo)
            ADAPTER_REPO="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --tasks TASKS        Evaluation tasks, comma-separated (default: arc_easy)"
            echo "  --limit NUM          Limit samples per task (optional)"
            echo "  --adapter_repo REPO  HuggingFace adapter repo (optional)"
            echo "  --device DEVICE      Device to use (default: cuda:0)"
            echo "  --dtype DTYPE        Data type (default: bfloat16)"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Available tasks:"
            echo "  arc_easy, arc_challenge, hellaswag, winogrande, piqa, boolq"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo -e "${CYAN}=============================================="
echo "GPT-OSS MixLoRA - Evaluation"
echo "==============================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  📋 Tasks: $TASKS"
echo "  🔢 Limit: ${LIMIT:-'all'}"
echo "  📦 Adapter: ${ADAPTER_REPO:-'default from env'}"
echo "  🖥️  Device: $DEVICE"
echo "  📊 Dtype: $DTYPE"
echo ""

# Build command
CMD="python run.py evaluate"
CMD="$CMD --tasks $TASKS"
CMD="$CMD --device $DEVICE"
CMD="$CMD --dtype $DTYPE"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ -n "$ADAPTER_REPO" ]; then
    CMD="$CMD --adapter_repo $ADAPTER_REPO"
fi

echo -e "${YELLOW}Running: $CMD${NC}"
echo ""

# Execute
eval $CMD

echo -e "\n${GREEN}✅ Evaluation complete!${NC}"
