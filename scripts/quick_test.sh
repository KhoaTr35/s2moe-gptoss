#!/bin/bash
# =============================================================================
# Quick Test Script - Minimal run to verify everything works
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

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo -e "${CYAN}=============================================="
echo "GPT-OSS MixLoRA - Quick Test"
echo "==============================================${NC}"
echo ""
echo "This script runs minimal tests to verify the setup works."
echo ""

# Check environment
echo -e "${YELLOW}  Checking environment...${NC}"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo -e "\n${YELLOW}  Checking imports...${NC}"
python3 -c "
from src.pipeline import Pipeline
from configs.train_config import TrainConfig
from configs.env_config import EnvConfig
from configs.data_config import DataConfig
from src.model import load_base_model, load_tokenizer
from mixlora import MixLoraConfig
print('  ✅ All imports successful')
"

echo -e "\n${YELLOW}  Testing training (10 steps)...${NC}"
python run.py train \
    --num_samples 100 \
    --max_steps 10 \
    --batch_size 1 \
    --no_wandb

echo -e "\n${YELLOW}  Testing inference...${NC}"
python run.py infer --text "Hello, I am"

echo -e "\n${GREEN}=============================================="
echo "✅ All quick tests passed!"
echo "==============================================${NC}"
echo ""
echo "The project is correctly set up and working."
echo ""
echo "Next steps:"
echo "  - Full training:  ./scripts/train.sh --num_samples 17000 --push_to_hub"
echo "  - Evaluation:     ./scripts/evaluate.sh --tasks arc_easy --limit 100"
echo "  - Full pipeline:  ./scripts/full_pipeline.sh --push_to_hub"
