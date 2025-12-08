"""
Logging utilities for GPT-OSS MixLoRA.
"""
import os
import logging
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to log to
        log_format: Custom log format
        
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger = logging.getLogger("mixlora-gptoss")
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def setup_wandb(
    project: str = "mixlora-gptoss",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
    tags: Optional[list] = None,
):
    """
    Setup Weights & Biases logging.
    
    Args:
        project: W&B project name
        entity: W&B entity/team name
        run_name: Name for this run
        config: Configuration dict to log
        tags: Tags for the run
        
    Returns:
        W&B run object
    """
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping W&B setup.")
        return None
    
    # Check for API key
    if not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set. W&B logging may fail.")
    
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"mixlora-gptoss-{timestamp}"
    
    # Initialize W&B
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        tags=tags,
        reinit=True,
    )
    
    return run


def log_metrics(metrics: dict, step: Optional[int] = None):
    """
    Log metrics to W&B if available.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def log_model_info(model, config: dict):
    """
    Log model information.
    
    Args:
        model: The model to log info for
        config: Configuration dict
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params,
        **config,
    }
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    log_metrics({"model_info": info})
