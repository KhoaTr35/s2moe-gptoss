"""
Training configuration for GPT-OSS MixLoRA.
Contains hyperparameters, batch sizes, learning rates, etc.
"""
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class MixLoraHyperParams:
    """MixLoRA specific hyperparameters."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    num_experts: int = 4
    top_k: int = 2
    router_init_range: float = 0.02
    jitter_noise: float = 0.0
    router_aux_loss_coef: float = 0.001
    
    # Augmentation parameters
    use_augmentation: bool = True
    alpha_scale: float = 1.0
    beta_scale: float = 1.0
    
    # Target modules for GPT-OSS/SwiGLU architecture
    target_modules: Dict[str, bool] = field(default_factory=lambda: {
        "q_proj": True,
        "k_proj": True,
        "v_proj": True,
        "o_proj": True,
        "gate_proj": True,
        "up_proj": True,
        "down_proj": True,
    })


@dataclass 
class TrainConfig:
    """Training configuration."""
    
    # Model
    model_id: str = "openai/gpt-oss-20b"
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True
    device_map: str = "auto"
    
    # MixLoRA
    mixlora: MixLoraHyperParams = field(default_factory=MixLoraHyperParams)
    
    # Training hyperparameters
    output_dir: str = "./mixlora_gptoss_output"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    max_steps: int = -1  # -1 for full epochs
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Optimization
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Reporting
    report_to: str = "wandb"
    run_name: Optional[str] = None
    
    # Misc
    remove_unused_columns: bool = False
    seed: int = 42
    
    def to_training_arguments_dict(self) -> dict:
        """Convert to HuggingFace TrainingArguments compatible dict."""
        return {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "optim": self.optim,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "remove_unused_columns": self.remove_unused_columns,
            "seed": self.seed,
        }
    
    @classmethod
    def for_testing(cls) -> "TrainConfig":
        """Create a minimal config for testing."""
        config = cls()
        config.max_steps = 10
        config.logging_steps = 5
        config.save_steps = 10
        config.report_to = "none"
        return config
    
    @classmethod
    def for_full_training(cls) -> "TrainConfig":
        """Create config for full training run."""
        config = cls()
        config.num_train_epochs = 1
        config.max_steps = -1
        return config
