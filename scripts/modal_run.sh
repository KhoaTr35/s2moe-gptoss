#!/bin/bash
# =============================================================================
# Modal Cloud Training Script
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

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default values
MODE=${MODE:-"train"}
NUM_SAMPLES=${NUM_SAMPLES:-17000}
EVAL_TASKS=${EVAL_TASKS:-"arc_easy"}
EVAL_LIMIT=${EVAL_LIMIT:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --eval-tasks)
            EVAL_TASKS="$2"
            shift 2
            ;;
        --eval-limit)
            EVAL_LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --mode MODE          Mode: train, evaluate, full (default: train)"
            echo "  --num-samples NUM    Number of training samples (default: 17000)"
            echo "  --eval-tasks TASKS   Evaluation tasks (default: arc_easy)"
            echo "  --eval-limit NUM     Limit eval samples (optional)"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}=============================================="
echo "GPT-OSS MixLoRA - Modal Cloud Run"
echo "==============================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Mode: $MODE"
echo "  Samples: $NUM_SAMPLES"
echo "  Eval Tasks: $EVAL_TASKS"
echo "  Eval Limit: ${EVAL_LIMIT:-'all'}"
echo ""

# Check Modal authentication
# echo -e "${YELLOW}Checking Modal authentication...${NC}"
# if ! modal token show > /dev/null 2>&1; then
#     echo -e "${RED}❌ Modal not authenticated. Run: modal token new${NC}"
#     exit 1
# fi
# echo -e "${GREEN}✅ Modal authenticated${NC}"
# echo ""

# Build Modal command
CMD="modal run src/modal/modal_entry.py"
CMD="$CMD --mode $MODE"

if [ "$MODE" = "train" ] || [ "$MODE" = "full" ]; then
    CMD="$CMD --num-samples $NUM_SAMPLES"
fi

if [ "$MODE" = "evaluate" ] || [ "$MODE" = "full" ]; then
    CMD="$CMD --eval-tasks $EVAL_TASKS"
    if [ -n "$EVAL_LIMIT" ]; then
        CMD="$CMD --eval-limit $EVAL_LIMIT"
    fi
fi

echo -e "${YELLOW}Running: $CMD${NC}"
echo ""

# Execute
eval $CMD

echo -e "\n${GREEN}✅ Modal run complete!${NC}"
