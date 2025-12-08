"""
Modal.com configuration for cloud GPU training.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModalConfig:
    """Modal.com deployment configuration."""
    
    # App settings
    app_name: str = "gptoss-mixlora-training"
    
    # GPU settings
    gpu_type: str = "H100"  # Use H100 (80GB) for large models like gpt-oss-20b
    gpu_count: int = 1
    
    # Container settings
    image_python_version: str = "3.11"
    timeout_seconds: int = 86400  # 24 hours
    
    # Volume settings
    volume_name: str = "gptoss-training-vol"
    volume_mount_path: str = "/vol"
    
    # Model cache
    model_cache_dir: str = "/vol/models"
    output_dir: str = "/vol/outputs"
    
    # Dependencies
    pip_packages: List[str] = field(default_factory=lambda: [
        "torch>=2.3.0",
        "transformers>=4.42.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "huggingface_hub>=0.19.0",
        "wandb>=0.16.0",
        "safetensors>=0.4.0",
    ])
    
    # Secrets - Modal secret names (existing secrets on Modal)
    # These secrets were created with:
    #   modal secret create huggingface-secret HF_TOKEN=hf_xxx
    #   modal secret create wandb-secret WANDB_API_KEY=xxx
    hf_secret_name: str = "huggingface-secret"
    wandb_secret_name: str = "wandb-secret"
    
    # Environment variable names (used inside the container)
    hf_token_env_var: str = "HF_TOKEN"
    wandb_api_key_env_var: str = "WANDB_API_KEY"
    
    # Memory
    memory_mb: int = 32768  # 32GB
    
    @property
    def gpu_config(self) -> str:
        """Get GPU config string for Modal."""
        return f"{self.gpu_type}:{self.gpu_count}"
    
    @classmethod
    def for_testing(cls) -> "ModalConfig":
        """Create a minimal config for testing."""
        config = cls()
        config.gpu_type = "T4"
        config.timeout_seconds = 3600  # 1 hour
        return config
    
    @classmethod
    def for_production(cls) -> "ModalConfig":
        """Create config for production training."""
        config = cls()
        config.gpu_type = "A100"
        config.gpu_count = 1
        config.timeout_seconds = 86400 * 2  # 48 hours
        return config
