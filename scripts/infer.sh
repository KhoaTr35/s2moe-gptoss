#!/bin/bash
# =============================================================================
# Inference Script for GPT-OSS MixLoRA
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
TEXT=${TEXT:-"Hello, I am a language model."}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.2}
ADAPTER_REPO=${ADAPTER_REPO:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --text)
            TEXT="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --repetition_penalty)
            REPETITION_PENALTY="$2"
            shift 2
            ;;
        --adapter_repo)
            ADAPTER_REPO="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --text TEXT                Input text prompt"
            echo "  --max_new_tokens NUM       Max new tokens to generate (default: 128)"
            echo "  --temperature TEMP         Temperature (default: 0.7)"
            echo "  --top_p P                  Top-p sampling (default: 0.9)"
            echo "  --repetition_penalty P     Repetition penalty (default: 1.2)"
            echo "  --adapter_repo REPO        HuggingFace adapter repo (optional)"
            echo "  --help, -h                 Show this help"
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
echo "GPT-OSS MixLoRA - Inference"
echo "==============================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  📝 Input: \"$TEXT\""
echo "  🔢 Max New Tokens: $MAX_NEW_TOKENS"
echo "  🌡️  Temperature: $TEMPERATURE"
echo "  📊 Top-p: $TOP_P"
echo "  🔄 Repetition Penalty: $REPETITION_PENALTY"
echo "  📦 Adapter: ${ADAPTER_REPO:-'default from env'}"
echo ""

# Build command
CMD="python run.py infer"
CMD="$CMD --text \"$TEXT\""
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --repetition_penalty $REPETITION_PENALTY"

if [ -n "$ADAPTER_REPO" ]; then
    CMD="$CMD --adapter_repo $ADAPTER_REPO"
fi

echo -e "${YELLOW}Running: $CMD${NC}"
echo ""

# Execute
eval $CMD

echo -e "\n${GREEN}✅ Inference complete!${NC}"
