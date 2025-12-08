"""
Dataset configuration for GPT-OSS MixLoRA training.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Dataset configuration."""
    
    # Dataset source
    dataset_name: str = "zwhe99/commonsense_170k"
    dataset_split: str = "train"
    
    # Processing
    num_samples: int = 170000
    max_length: int = 1024
    
    # Tokenization
    truncation: bool = True
    padding: str = "max_length"
    
    # Batching for map operations
    map_batch_size: int = 1000
    
    # Prompt format
    prompt_template: str = "alpaca"  # Options: alpaca, chatml, raw
    
    # Cache
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    @classmethod
    def for_testing(cls, num_samples: int = 1000) -> "DataConfig":
        """Create a minimal config for testing."""
        config = cls()
        config.num_samples = num_samples
        config.max_length = 512
        return config
    
    @classmethod  
    def for_full_training(cls) -> "DataConfig":
        """Create config for full training."""
        return cls()
