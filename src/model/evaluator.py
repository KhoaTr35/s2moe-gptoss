"""
Evaluation module for GPT-OSS MixLoRA using lm-evaluation-harness.
"""
import os
import gc
import shutil
import subprocess
import torch
from typing import Optional, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from mixlora import load_adapter_weights
from configs.env_config import EnvConfig
from configs.train_config import TrainConfig
from .load_model import inject_mixlora_into_gptoss


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def prepare_model_for_eval(
    model_id: str = "openai/gpt-oss-20b",
    adapter_repo_id: str = "twanghcmut/mixlora-gpt-oss-experimental-run",
    device: str = "cuda",
    target_gpu: str = "cuda:0",
    eval_dtype: torch.dtype = torch.bfloat16,
) -> PreTrainedModel:
    """
    Load and prepare model with MixLoRA adapter for evaluation.
    
    Args:
        model_id: Base model ID
        adapter_repo_id: HuggingFace repo with adapter
        device: Device to use
        target_gpu: Target GPU
        eval_dtype: Data type for evaluation
        
    Returns:
        Model ready for evaluation
    """
    print(f"Loading base model with dtype={eval_dtype} on {target_gpu}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=eval_dtype,
        device_map=target_gpu,
        trust_remote_code=True,
    )
    
    print(f"Loading MixLoRA adapter weights with dtype={eval_dtype}...")
    loaded_config, loaded_weights = load_adapter_weights(
        adapter_repo_id,
        adapter_name="default",
        device=device,
        dtype=eval_dtype,
    )
    
    print("Injecting adapter into base model...")
    model = inject_mixlora_into_gptoss(model, loaded_config, loaded_weights)
    model.eval()
    
    return model


def save_combined_model(
    model: PreTrainedModel,
    tokenizer,
    save_dir: str = "mixlora_full_model_for_eval",
):
    """
    Save combined model (base + adapter) for lm-eval.
    
    Args:
        model: Model with adapter injected
        tokenizer: Tokenizer
        save_dir: Directory to save combined model
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    print(f"Saving combined model to {save_dir}...")
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Combined model saved to {save_dir}")


def run_lm_eval(
    model_path: str,
    tasks: List[str] = None,
    device: str = "cuda:0",
    limit: Optional[int] = None,
    batch_size: int = 8,
    dtype: str = "bfloat16",
) -> str:
    """
    Run lm-evaluation-harness on the model.
    
    Args:
        model_path: Path to saved model
        tasks: List of evaluation tasks
        device: Device to use
        limit: Number of samples per task (None for full)
        batch_size: Batch size
        dtype: Data type string
        
    Returns:
        Command output
    """
    if tasks is None:
        tasks = ["arc_easy"]
    
    tasks_str = ",".join(tasks)
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},tokenizer={model_path},dtype={dtype}",
        "--tasks", tasks_str,
        "--device", device,
        "--batch_size", str(batch_size),
    ]
    
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running lm_eval: {result.stderr}")
    else:
        print(result.stdout)
    
    return result.stdout


def evaluate_model(
    model_id: str = "openai/gpt-oss-20b",
    adapter_repo_id: Optional[str] = None,
    env_config: Optional[EnvConfig] = None,
    tasks: List[str] = None,
    limit: Optional[int] = None,
    temp_save_dir: str = "mixlora_full_model_for_eval",
    cleanup_after: bool = True,
) -> str:
    """
    Full evaluation pipeline for GPT-OSS MixLoRA.
    
    Args:
        model_id: Base model ID
        adapter_repo_id: Adapter repo ID (uses env_config if None)
        env_config: Environment configuration
        tasks: Evaluation tasks
        limit: Sample limit per task
        temp_save_dir: Temporary directory for combined model
        cleanup_after: Whether to cleanup temp files
        
    Returns:
        Evaluation results
    """
    if env_config is None:
        env_config = EnvConfig.from_env()
    
    if adapter_repo_id is None:
        adapter_repo_id = env_config.full_repo_id
    
    if tasks is None:
        tasks = ["arc_easy", "arc_challenge", "hellaswag"]
    
    device = env_config.device
    target_gpu = env_config.target_gpu
    eval_dtype = torch.bfloat16
    
    # Load and prepare model
    print("\n--- Loading model for evaluation ---")
    model = prepare_model_for_eval(
        model_id=model_id,
        adapter_repo_id=adapter_repo_id,
        device=device,
        target_gpu=target_gpu,
        eval_dtype=eval_dtype,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Save combined model
    save_combined_model(model, tokenizer, temp_save_dir)
    
    # Cleanup GPU memory
    del model
    cleanup_memory()
    
    # Run evaluation
    print("\n--- Running lm-eval ---")
    results = run_lm_eval(
        model_path=temp_save_dir,
        tasks=tasks,
        device=target_gpu,
        limit=limit,
    )
    
    # Cleanup temp files
    if cleanup_after and os.path.exists(temp_save_dir):
        shutil.rmtree(temp_save_dir)
        print(f"✅ Cleaned up {temp_save_dir}")
    
    return results
