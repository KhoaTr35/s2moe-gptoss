"""
Path utilities for GPT-OSS MixLoRA.
"""
import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Navigate up from this file to find project root
    current = Path(__file__).resolve()
    
    # Go up until we find a directory with known markers
    markers = ["run.py", "requirements.txt", "setup.py", "mixlora"]
    
    for parent in current.parents:
        for marker in markers:
            if (parent / marker).exists():
                return parent
    
    # Fallback to current working directory
    return Path.cwd()


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_output_dir(name: Optional[str] = None) -> Path:
    """
    Get output directory for training artifacts.
    
    Args:
        name: Optional subdirectory name
        
    Returns:
        Path to output directory
    """
    root = get_project_root()
    output_dir = root / "outputs"
    
    if name:
        output_dir = output_dir / name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_cache_dir() -> Path:
    """
    Get cache directory for downloaded models/datasets.
    
    Returns:
        Path to cache directory
    """
    root = get_project_root()
    cache_dir = root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_config_path(config_name: str) -> Path:
    """
    Get path to a configuration file.
    
    Args:
        config_name: Name of config file (without extension)
        
    Returns:
        Path to config file
    """
    root = get_project_root()
    
    # Try different extensions
    for ext in [".yaml", ".yml", ".json", ".py"]:
        path = root / "configs" / f"{config_name}{ext}"
        if path.exists():
            return path
    
    # Return default path
    return root / "configs" / f"{config_name}.yaml"


def get_model_path(model_name: str) -> Path:
    """
    Get path for saved model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path for model storage
    """
    return get_output_dir("models") / model_name


def get_adapter_path(adapter_name: str = "default") -> Path:
    """
    Get path for saved adapter.
    
    Args:
        adapter_name: Name of the adapter
        
    Returns:
        Path for adapter storage
    """
    return get_output_dir("adapters") / adapter_name
