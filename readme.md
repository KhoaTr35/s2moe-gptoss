# README

Project Refactor Instructions for AI Agent
Integrating `gpt-oss.ipynb` Into a Modular Python Project


## 1. Overview

This document provides a complete execution plan for transforming the existing `gpt-oss.ipynb` notebook—responsible for training and evaluating a MixLoRA-based model on Modal.com into a properly structured Python package.
The agent must reorganize the project into modules while **preserving all original logic, math, training flow, hyperparameters, and implementation details**.

Only the structure, directory layout, and execution entrypoints may be changed.
Any code modifications must be minimal and strictly related to modularization.


## 2. High-Level Goals

The agent must:

1. **Extract all code from `gpt-oss.ipynb`** and convert it into a clean Python project.
2. **Preserve all original functionality**, including:

   * Modal.com GPU + containers integration
   * Modal Volumes setup
   * HuggingFace model loading using `hf_token`
   * W&B logging
   * MixLoRA components in the `mixlora` folder
   * Training, evaluation, dataset prep, inference, metrics
3. Create a **single unified entrypoint script** (e.g., `run.py`) that executes the full pipeline with minimal user steps.
4. **Do not modify algorithms, configs, or behavior**, unless required to adapt to the modular structure.
5. Provide a clean, modern **AI experiment project layout** similar to contemporary LLM training frameworks.
6. Add documentation and CLI usability.
7. Ensure the final structure is stable for:

   * reproducible training runs
   * batch experiments
   * scaling to multiple configurations


## 3. Target Project Structure

The agent must restructure the project into the following layout:

```
project-root/
│
├── mixlora/                 # unchanged MixLoRA repo code
│   ├── prompter.py
│   ├── lora.py
│   ├── utils.py
│   └── ...
│
├── configs/
│   ├── modal_config.py      # modal client, image, volume setup
│   ├── train_config.py      # hyperparameters, batch sizes, LR, etc.
│   ├── data_config.py       # dataset paths (HF or local)
│   └── env_config.py        # env variables: hf_token, wandb_key
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataloader.py
│   │   ├── preprocess.py
│   │   └── dataset_utils.py
│   │
│   ├── model/
│   │   ├── load_model.py     # HF model/Tokenizer + LoRA injection
│   │   ├── trainer.py        # training loop
│   │   ├── evaluator.py      # evaluation loop
│   │   └── infer.py          # inference helpers
│   │
│   ├── modal/
│   │   ├── modal_entry.py    # modal function entrypoints
│   │   ├── modal_setup.py    # container + volume definitions
│   │   └── modal_utils.py
│   │
│   ├── utils/
│   │   ├── logging.py        # wandb setup
│   │   ├── timer.py
│   │   ├── paths.py
│   │   └── io.py
│   │
│   └── pipeline.py           # orchestrates training → eval → upload
│
├── run.py                    # single entry point for the entire project
│
├── requirements.txt
├── README.md
└── setup.py                  # optional
```

**Important:**
The agent MUST ensure code contents from the notebook are placed in the correct module while keeping the logic intact.


## 4. Required Tasks for the AI Agent

### 4.1 Convert Notebook Cells Into Modules

The agent must:

1. Parse each notebook cell sequentially.
2. Group related sections into Python modules under `src/`.
3. Move MixLoRA utilities into logically grouped files in `mixlora/` without changing content.
4. Extract all configuration values into `configs/`.

**Do not rewrite algorithms, function bodies, or change hyperparameters.**


### 4.2 Build a Unified Execution Pipeline

The notebook currently runs training cell-by-cell.
The agent must convert this into:

A central orchestrator:

```
src/pipeline.py
```

This orchestrator must sequentially:

1. Load configs
2. Initialize Modal resources
3. Load HF model + tokenizer
4. Apply MixLoRA components
5. Prepare datasets
6. Train the model
7. Evaluate
8. Log results to WandB
9. Save and upload artifacts
10. Shut down cleanly

All internal calls should directly reuse the notebook logic.


### 4.3 Create Modal Entry Scripts

Modal functions currently defined directly in the notebook must become Python functions:

`src/modal/modal_entry.py`

This includes:

* container image builder
* volumes mount
* remote functions for training
* remote functions for evaluation
* any remote inference code

Logic remains identical.


### 4.4 Add a Single Command Entry

`run.py` must allow running the full pipeline simply as:

```
python run.py --config configs/train_config.py
```

or

```
python run.py train
python run.py evaluate
python run.py inference --text "..."
```

This gives a clean workflow for experiments.


### 4.5 Add Minimal Tooling

Agent must generate:

* `requirements.txt`
* `.gitignore`
* README.md (for human users)
* prepare `wandb` initialization
* environment loading helpers


### 4.6 Ensure End-to-End Reproducibility

Agent must ensure:

* same seeds
* same tokenizer loading logic
* same dataset splits
* same training sequence
* same Modal container behavior
* same HF model loading order
* same W&B logs

Nothing should change in the computational behavior.


## 5. Constraints the Agent Must Respect

**ABSOLUTE RULES:**

1. **Do NOT modify model weights logic, LoRA behavior, or MixLoRA internals.**
2. **Do NOT change dataset logic, training steps, evaluation metrics, or WandB logs.**
3. **Do NOT rewrite modal container definitions except to move them.**
4. **Do NOT change API calls or request parameters to HF or WandB.**
5. **Only restructure — no algorithmic or logical alterations.**

Allowed minor edits:

* Rename variables for clarity if needed for modularization
* Wrap notebook sections into functions
* Add imports / argument parsing
* Adjust path references due to new project structure


## 6. Optimal Workflow for the Final Project

The agent must produce a project that can run with:

### **Step 1: Set environment variables**

```
export HF_TOKEN=...
export WANDB_API_KEY=...
```

### **Step 2: Run training**

```
python run.py train
```

### **Step 3: Evaluate**

```
python run.py evaluate
```

### **Step 4: Inference**

```
python run.py infer --text "Hello"
```

### **Step 5: Deploy on Modal (if needed)**

```
modal run src/modal/modal_entry.py::train_remote
```


## 7. Deliverables Expected from the AI Agent

The agent must output:

1. A full project folder containing all modules shown above.
2. All code extracted from the notebook into properly structured `.py` files.
3. A replacement README for the new project.
4. A working `run.py` executable.
5. Consistent imports throughout modules.
6. No broken paths or missing logic.


## 8. Additional Enhancements (Optional but Recommended)

The agent may:

* Add `pyproject.toml` if needed
* Add `configs/yaml/` option for experiments
* Add CLI with `argparse` or `typer`
* Add experiment versioning in `wandb`

But only if these changes do NOT alter implementation logic.


## 9. Completion Criteria

The refactor is considered successful when:

* The entire training pipeline runs identically to the notebook.
* Modal training works without modification of logic.
* HuggingFace model loads identically.
* Final results match previous W&B outputs.
* The new modular project can be executed with a single command.

---

# ✅ Refactoring Complete

## Project Structure Created

```
project-root/
│
├── mixlora/                 # MixLoRA implementation (preserved)
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── lora_linear.py
│   ├── prompter.py
│   └── utils.py
│
├── configs/                 # Configuration modules
│   ├── __init__.py
│   ├── modal_config.py      # Modal.com settings
│   ├── train_config.py      # Training hyperparameters
│   ├── data_config.py       # Dataset configuration
│   └── env_config.py        # Environment variables
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── preprocess.py
│   │   └── dataset_utils.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── load_model.py     # Model loading + MixLoRA injection
│   │   ├── trainer.py        # Training logic
│   │   ├── evaluator.py      # Evaluation pipeline
│   │   └── infer.py          # Inference utilities
│   │
│   ├── modal/
│   │   ├── __init__.py
│   │   ├── modal_entry.py    # Modal function entrypoints
│   │   ├── modal_setup.py    # Container + volume definitions
│   │   └── modal_utils.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py        # WandB setup
│   │   ├── timer.py          # Performance timing
│   │   ├── paths.py          # Path management
│   │   └── io.py             # I/O operations
│   │
│   └── pipeline.py           # Main orchestrator
│
├── run.py                    # CLI entry point
├── requirements.txt          # Dependencies
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## Usage Instructions

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Set Environment Variables

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
```

### Training

```bash
# Basic training
python run.py train --output_dir ./outputs/mixlora-gptoss

# Training with custom settings
python run.py train \
    --model_name openai/gpt-oss-20b \
    --num_experts 4 \
    --lora_r 8 \
    --num_epochs 3

# Training on Modal.com
python run.py train --use_modal --gpu_type h100
```

### Evaluation

```bash
python run.py evaluate \
    --checkpoint_path ./outputs/mixlora-gptoss \
    --eval_tasks arc_easy,hellaswag
```

### Inference

```bash
python run.py infer \
    --checkpoint_path ./outputs/mixlora-gptoss \
    --prompt "What is the capital of France?"
```

### Modal.com Deployment

```bash
# Run training on Modal
modal run src/modal/modal_entry.py::train_on_modal

# Monitor logs
modal logs -f
```

## Key Features

- **MixLoRA Architecture**: Mixture of LoRA experts with learnable routing
- **Dual-Path Augmentation**: NormalAugmenter for enhanced training
- **Modal.com Integration**: Cloud GPU training (H100/A100)
- **Comprehensive Evaluation**: lm-evaluation-harness support
- **HuggingFace Integration**: Easy model sharing

## Configuration

### Training Config (`configs/train_config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `openai/gpt-oss-20b` | Base model |
| `num_experts` | 4 | Number of LoRA experts |
| `top_k` | 2 | Top-k routing |
| `lora_r` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling |
| `num_epochs` | 3 | Training epochs |
| `batch_size` | 4 | Batch size |
| `learning_rate` | 5e-5 | Learning rate |
