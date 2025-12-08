#!/bin/bash
# =============================================================================
# Training Script for GPT-OSS MixLoRA
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
LORA_R=${LORA_R:-8}
NUM_EXPERTS=${NUM_EXPERTS:-4}
TOP_K=${TOP_K:-2}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}
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
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --num_experts)
            NUM_EXPERTS="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --push_to_hub)
            PUSH_TO_HUB=true
            shift
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
            echo "  --lora_r R           LoRA rank (default: 8)"
            echo "  --num_experts N      Number of experts (default: 4)"
            echo "  --top_k K            Top-K experts (default: 2)"
            echo "  --output_dir DIR     Output directory (default: ./outputs)"
            echo "  --push_to_hub        Push to HuggingFace Hub"
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
echo "GPT-OSS MixLoRA - Training"
echo "==============================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  📊 Samples: $NUM_SAMPLES"
echo "  🔢 Max Steps: ${MAX_STEPS:-'auto'}"
echo "  📦 Batch Size: $BATCH_SIZE"
echo "  📈 Learning Rate: $LEARNING_RATE"
echo "  🔧 LoRA Rank: $LORA_R"
echo "  👥 Experts: $NUM_EXPERTS (Top-$TOP_K)"
echo "  📁 Output: $OUTPUT_DIR"
echo "  ☁️  Push to Hub: $PUSH_TO_HUB"
echo "  📊 WandB: $USE_WANDB"
echo ""

# Build command
CMD="python run.py train"
CMD="$CMD --num_samples $NUM_SAMPLES"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --lora_r $LORA_R"
CMD="$CMD --num_experts $NUM_EXPERTS"
CMD="$CMD --top_k $TOP_K"
CMD="$CMD --output_dir $OUTPUT_DIR"

if [ -n "$MAX_STEPS" ]; then
    CMD="$CMD --max_steps $MAX_STEPS"
fi

if [ "$PUSH_TO_HUB" = true ]; then
    CMD="$CMD --push_to_hub"
fi

if [ "$USE_WANDB" = false ]; then
    CMD="$CMD --no_wandb"
fi

echo -e "${YELLOW}Running: $CMD${NC}"
echo ""

# Execute
eval $CMD

echo -e "\n${GREEN}✅ Training complete!${NC}"
