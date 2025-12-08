#!/bin/bash
# =============================================================================
# Setup Script for GPT-OSS MixLoRA
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "GPT-OSS MixLoRA - Environment Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo -e "${YELLOW}📁 Project directory: $PROJECT_DIR${NC}"

# Check Python version
echo -e "\n${YELLOW}🐍 Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo -e "\n${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "\n${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}🔄 Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "\n${YELLOW}📥 Installing requirements...${NC}"
pip install -r requirements.txt

# Check CUDA availability
echo -e "\n${YELLOW}🎮 Checking CUDA availability...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Check if environment variables are set
echo -e "\n${YELLOW}🔐 Checking environment variables...${NC}"
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}⚠️  HF_TOKEN is not set. Set it with: export HF_TOKEN='your_token'${NC}"
else
    echo -e "${GREEN}✅ HF_TOKEN is set${NC}"
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${YELLOW}ℹ️  WANDB_API_KEY is not set (optional for logging)${NC}"
else
    echo -e "${GREEN}✅ WANDB_API_KEY is set${NC}"
fi

echo -e "\n${GREEN}=============================================="
echo "✅ Setup complete!"
echo "=============================================="
echo -e "${NC}"
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To set required environment variables:"
echo "  export HF_TOKEN='your_huggingface_token'"
echo "  export WANDB_API_KEY='your_wandb_key'  # Optional"
