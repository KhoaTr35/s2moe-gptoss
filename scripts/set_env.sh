#!/bin/bash
# =============================================================================
# Set Environment Variables Script
# Source this file to set up environment variables
# Usage: source scripts/set_env.sh
# =============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get project directory (handle both sourcing and direct execution)
if [[ -n "${BASH_SOURCE[0]}" && "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # Script is being sourced
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    # Script is being executed directly
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# If SCRIPT_DIR is just "scripts", resolve relative to pwd
if [[ "$SCRIPT_DIR" == "." || "$SCRIPT_DIR" == "scripts" || ! -d "$SCRIPT_DIR" ]]; then
    SCRIPT_DIR="$(pwd)/scripts"
fi

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Fallback: if .env not found, try current directory
if [[ ! -f "$PROJECT_DIR/.env" && -f ".env" ]]; then
    PROJECT_DIR="$(pwd)"
fi

echo -e "${YELLOW}Setting environment variables...${NC}"

# Load from .env file if exists
ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}✅ Loading from .env file${NC}"
    # Export all variables from .env (ignore comments and empty lines)
    set -a
    source "$ENV_FILE"
    set +a
else
    echo -e "${RED}⚠️  .env file not found at $ENV_FILE${NC}"
    echo -e "   Create it with: cp .env.example .env"
fi

# Verify required variables
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}❌ HF_TOKEN is not set!${NC}"
    echo "   Please add it to your .env file"
else
    echo -e "${GREEN}✅ HF_TOKEN is set${NC}"
fi

# Optional variables info
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${YELLOW}ℹ️  WANDB_API_KEY not set (optional)${NC}"
else
    echo -e "${GREEN}✅ WANDB_API_KEY is set${NC}"
fi

# Set defaults for optional variables
export HF_USERNAME="${HF_USERNAME:-twanghcmut}"
export HF_REPO_NAME="${HF_REPO_NAME:-mixlora-gpt-oss-experimental-run}"
export MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"
export WANDB_PROJECT="${WANDB_PROJECT:-mixlora-gptoss}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo ""
echo -e "${GREEN}Environment variables:${NC}"
echo "  HF_USERNAME: $HF_USERNAME"
echo "  HF_REPO_NAME: $HF_REPO_NAME"
echo "  MODEL_ID: $MODEL_ID"
echo "  WANDB_PROJECT: $WANDB_PROJECT"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""
