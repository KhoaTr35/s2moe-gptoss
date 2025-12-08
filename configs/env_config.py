"""
Environment configuration for GPT-OSS MixLoRA.
Handles environment variables for HuggingFace, WandB, and other services.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    
    # Find .env file (check project root)
    _project_root = Path(__file__).parent.parent
    _env_path = _project_root / ".env"
    
    if _env_path.exists():
        load_dotenv(_env_path)
        print(f"✅ Loaded environment from {_env_path}")
    else:
        # Try current directory
        _env_path = Path(".env")
        if _env_path.exists():
            load_dotenv(_env_path)
            print(f"✅ Loaded environment from {_env_path}")
        else:
            print("⚠️  No .env file found, using system environment variables")
            
except ImportError:
    print("⚠️  python-dotenv not installed. Run: pip install python-dotenv")


@dataclass
class EnvConfig:
    """Environment configuration for API keys and tokens."""
    
    # HuggingFace Hub
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN", None))
    hf_username: str = field(default_factory=lambda: os.environ.get("HF_USERNAME", "twanghcmut"))
    hf_repo_name: str = field(default_factory=lambda: os.environ.get("HF_REPO_NAME", "mixlora-gpt-oss-experimental-run")) # đổi tên hf repo tránh đè
    
    # Weights & Biases
    wandb_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("WANDB_API_KEY", None))
    wandb_project: str = field(default_factory=lambda: os.environ.get("WANDB_PROJECT", "mixlora-gptoss"))
    wandb_entity: Optional[str] = field(default_factory=lambda: os.environ.get("WANDB_ENTITY", None))
    
    # Device settings
    device: str = field(default_factory=lambda: os.environ.get("DEVICE", "cuda"))
    target_gpu: str = field(default_factory=lambda: os.environ.get("TARGET_GPU", "cuda:0"))
    
    @property
    def full_repo_id(self) -> str:
        """Get full HuggingFace repo ID."""
        return f"{self.hf_username}/{self.hf_repo_name}"
    
    def validate(self) -> bool:
        """Validate that required environment variables are set."""
        if self.hf_token is None:
            print("⚠️  Warning: HF_TOKEN not set. Hub operations may fail.")
            return False
        return True
    
    def setup_environment(self):
        """Setup environment variables for training."""
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
        
        if self.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.wandb_api_key
        
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        
        if self.wandb_entity:
            os.environ["WANDB_ENTITY"] = self.wandb_entity
    
    @classmethod
    def from_env(cls) -> "EnvConfig":
        """Create config from environment variables."""
        return cls()
    
    def __repr__(self) -> str:
        # Hide sensitive tokens
        hf_display = "***" if self.hf_token else "None"
        wandb_display = "***" if self.wandb_api_key else "None"
        return (
            f"EnvConfig(\n"
            f"  hf_token={hf_display},\n"
            f"  hf_username='{self.hf_username}',\n"
            f"  hf_repo_name='{self.hf_repo_name}',\n"
            f"  wandb_api_key={wandb_display},\n"
            f"  wandb_project='{self.wandb_project}',\n"
            f"  device='{self.device}',\n"
            f"  target_gpu='{self.target_gpu}'\n"
            f")"
        )
