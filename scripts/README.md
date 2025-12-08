# GPT-OSS MixLoRA Scripts

This folder contains shell scripts to run various operations for the GPT-OSS MixLoRA project.

## Quick Start

```bash
# 1. Setup environment (first time only)
./scripts/setup.sh

# 2. Set environment variables
source scripts/set_env.sh
export HF_TOKEN="your_huggingface_token"

# 3. Run quick test to verify setup
./scripts/quick_test.sh
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `setup.sh` | Install dependencies and create virtual environment |
| `set_env.sh` | Set environment variables (source this file) |
| `train.sh` | Run training |
| `evaluate.sh` | Run evaluation with lm-eval-harness |
| `infer.sh` | Run inference/generation |
| `full_pipeline.sh` | Run full pipeline (train + evaluate) |
| `quick_test.sh` | Quick test to verify everything works |
| `modal_run.sh` | Run on Modal.com cloud GPUs |

## Usage Examples

### Setup

```bash
# Initial setup
./scripts/setup.sh

# Set environment variables
source scripts/set_env.sh
export HF_TOKEN="hf_xxxxxxxxxxxx"
export WANDB_API_KEY="xxxxxxxxxxxxx"  # Optional
```
```bash
# Modal login
modal token new
```
### Training

```bash
# Full training (17k samples)
./scripts/train.sh --num_samples 17000 --push_to_hub

# Quick training (for testing)
./scripts/train.sh --num_samples 1000 --max_steps 100 --no_wandb

# Custom training
./scripts/train.sh \
    --num_samples 5000 \
    --max_steps 500 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --lora_r 8 \
    --num_experts 4 \
    --top_k 2 \
    --push_to_hub
```

### Evaluation

```bash
# Basic evaluation
./scripts/evaluate.sh --tasks arc_easy

# Quick evaluation (limited samples)
./scripts/evaluate.sh --tasks arc_easy --limit 100

# Multiple tasks
./scripts/evaluate.sh --tasks arc_easy,arc_challenge,hellaswag --limit 100

# From specific adapter
./scripts/evaluate.sh --adapter_repo username/my-adapter --tasks arc_easy
```

### Inference

```bash
# Basic inference
./scripts/infer.sh --text "What is the capital of France?"

# Custom generation settings
./scripts/infer.sh \
    --text "Explain quantum computing" \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --top_p 0.9
```

### Full Pipeline

```bash
# Full pipeline with evaluation
./scripts/full_pipeline.sh --num_samples 17000 --push_to_hub

# Quick test pipeline
./scripts/full_pipeline.sh \
    --num_samples 1000 \
    --max_steps 100 \
    --eval_tasks arc_easy \
    --eval_limit 10 \
    --no_wandb
```

### Modal Cloud

```bash
# Training on Modal
./scripts/modal_run.sh --mode train --num_samples 17000

# Evaluation on Modal
./scripts/modal_run.sh --mode evaluate --eval_tasks arc_easy

# Full pipeline on Modal
./scripts/modal_run.sh --mode full --num_samples 17000
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token |
| `WANDB_API_KEY` | No | Weights & Biases API key |
| `HF_USERNAME` | No | HuggingFace username (default: twanghcmut) |
| `HF_REPO_NAME` | No | Repository name (default: mixlora-gpt-oss-experimental) |
| `MODEL_ID` | No | Base model ID (default: openai/gpt-oss-20b) |
| `CUDA_VISIBLE_DEVICES` | No | GPU devices to use (default: 0) |

## Help

Each script supports `--help` for detailed options:

```bash
./scripts/train.sh --help
./scripts/evaluate.sh --help
./scripts/infer.sh --help
./scripts/full_pipeline.sh --help
./scripts/modal_run.sh --help
```
