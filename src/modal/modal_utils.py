"""
Modal.com utility functions.
"""
import os
from typing import Dict, Optional

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


def get_secrets(secret_names: list = None) -> Dict[str, str]:
    """
    Get secrets from Modal or environment.
    
    Args:
        secret_names: List of secret names to retrieve
        
    Returns:
        Dictionary of secret values
    """
    if secret_names is None:
        secret_names = ["HF_TOKEN", "WANDB_API_KEY"]
    
    secrets = {}
    for name in secret_names:
        value = os.environ.get(name)
        if value:
            secrets[name] = value
    
    return secrets


def setup_modal_environment():
    """
    Setup environment for Modal execution.
    Sets common environment variables and paths.
    """
    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set HuggingFace cache
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/vol/cache/huggingface"
    
    # Set transformers cache
    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = "/vol/cache/transformers"


def check_modal_available() -> bool:
    """Check if Modal is available."""
    return MODAL_AVAILABLE


def get_volume_path(subpath: str = "", base_path: str = "/vol") -> str:
    """
    Get path within Modal volume.
    
    Args:
        subpath: Subpath within volume
        base_path: Volume mount path
        
    Returns:
        Full path
    """
    return os.path.join(base_path, subpath)


def ensure_dir_exists(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def cleanup_gpu_memory():
    """Clean up GPU memory in Modal environment."""
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except ImportError:
        pass
