#!/bin/bash
# =============================================================================
# Full Pipeline Script (Train + Evaluate) for GPT-OSS MixLoRA
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
NUM_SAMPLES=${NUM_SAMPLES:-17000}
MAX_STEPS=${MAX_STEPS:-""}
BATCH_SIZE=${BATCH_SIZE:-2}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}
RUN_EVAL=${RUN_EVAL:-true}
EVAL_TASKS=${EVAL_TASKS:-"arc_easy"}
EVAL_LIMIT=${EVAL_LIMIT:-""}
USE_WANDB=${USE_WANDB:-true}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --push_to_hub)
            PUSH_TO_HUB=true
            shift
            ;;
        --no_eval)
            RUN_EVAL=false
            shift
            ;;
        --eval_tasks)
            EVAL_TASKS="$2"
            shift 2
            ;;
        --eval_limit)
            EVAL_LIMIT="$2"
            shift 2
            ;;
        --no_wandb)
            USE_WANDB=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --num_samples NUM    Number of training samples (default: 17000)"
            echo "  --max_steps NUM      Max training steps (optional)"
            echo "  --batch_size NUM     Batch size (default: 2)"
            echo "  --learning_rate LR   Learning rate (default: 2e-4)"
            echo "  --push_to_hub        Push to HuggingFace Hub"
            echo "  --no_eval            Skip evaluation"
            echo "  --eval_tasks TASKS   Evaluation tasks (default: arc_easy)"
            echo "  --eval_limit NUM     Limit eval samples (optional)"
            echo "  --no_wandb           Disable WandB logging"
            echo "  --help, -h           Show this help"
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
echo "GPT-OSS MixLoRA - Full Pipeline"
echo "==============================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Samples: $NUM_SAMPLES"
echo "  Max Steps: ${MAX_STEPS:-'auto'}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Push to Hub: $PUSH_TO_HUB"
echo "  Run Eval: $RUN_EVAL"
echo "  Eval Tasks: $EVAL_TASKS"
echo "  Eval Limit: ${EVAL_LIMIT:-'all'}"
echo "  WandB: $USE_WANDB"
echo ""

# Build command
CMD="python run.py full"
CMD="$CMD --num_samples $NUM_SAMPLES"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"

if [ -n "$MAX_STEPS" ]; then
    CMD="$CMD --max_steps $MAX_STEPS"
fi

if [ "$PUSH_TO_HUB" = true ]; then
    CMD="$CMD --push_to_hub"
fi

if [ "$RUN_EVAL" = false ]; then
    CMD="$CMD --no_eval"
else
    CMD="$CMD --eval_tasks $EVAL_TASKS"
    if [ -n "$EVAL_LIMIT" ]; then
        CMD="$CMD --eval_limit $EVAL_LIMIT"
    fi
fi

if [ "$USE_WANDB" = false ]; then
    CMD="$CMD --no_wandb"
fi

echo -e "${YELLOW}Running: $CMD${NC}"
echo ""

# Execute
eval $CMD

echo -e "\n${GREEN}=============================================="
echo "Full pipeline complete!"
echo "==============================================${NC}"
