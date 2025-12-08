"""
I/O utilities for GPT-OSS MixLoRA.
"""
import os
import json
from typing import Any, Dict, Optional
from pathlib import Path


def save_json(data: Dict[str, Any], path: str, indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output file path
        indent: JSON indentation
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def save_text(text: str, path: str):
    """
    Save text to file.
    
    Args:
        text: Text content
        path: Output file path
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    with open(path, "w") as f:
        f.write(text)


def load_text(path: str) -> str:
    """
    Load text from file.
    
    Args:
        path: Input file path
        
    Returns:
        File contents as string
    """
    with open(path, "r") as f:
        return f.read()


def file_exists(path: str) -> bool:
    """Check if file exists."""
    return os.path.exists(path)


def get_file_size(path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(path)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def list_files(directory: str, pattern: str = "*") -> list:
    """
    List files in directory matching pattern.
    
    Args:
        directory: Directory path
        pattern: Glob pattern
        
    Returns:
        List of file paths
    """
    path = Path(directory)
    return [str(f) for f in path.glob(pattern) if f.is_file()]


def copy_file(src: str, dst: str):
    """
    Copy file from src to dst.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    import shutil
    os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else ".", exist_ok=True)
    shutil.copy2(src, dst)


def remove_file(path: str):
    """
    Remove file if exists.
    
    Args:
        path: File path to remove
    """
    if os.path.exists(path):
        os.remove(path)


def remove_dir(path: str):
    """
    Remove directory and contents.
    
    Args:
        path: Directory path to remove
    """
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
