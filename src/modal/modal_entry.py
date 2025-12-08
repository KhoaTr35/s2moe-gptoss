"""
Modal.com entry points for remote training and evaluation.
"""
import os
import sys
from pathlib import Path

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# Get project root directory (for local imports only)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Only define Modal functions if Modal is available
if MODAL_AVAILABLE:
    # =========================================================================
    # IMPORTANT: Do NOT import any project modules here!
    # Modal copies this file to /root/modal_entry.py in the container,
    # but project files are mounted to /root/project/.
    # All project imports must happen INSIDE function bodies after sys.path setup.
    # =========================================================================
    
    # Hardcoded config values for Modal setup (mirrors configs/modal_config.py)
    APP_NAME = "gptoss-mixlora-training"
    GPU_TYPE = "H100"  # Use H100 (80GB) for large models like gpt-oss-20b
    VOLUME_NAME = "gptoss-training-vol"
    VOLUME_MOUNT_PATH = "/vol"
    TIMEOUT_SECONDS = 86400  # 24 hours
    PYTHON_VERSION = "3.11"
    HF_SECRET_NAME = "huggingface-secret"
    WANDB_SECRET_NAME = "wandb-secret"
    
    # Create app and resources
    app = modal.App(name=APP_NAME)
    
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    
    # Create image with project files included
    image = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(
            "torch>=2.3.0",
            "transformers>=4.42.0",
            "datasets>=2.16.0",
            "accelerate>=0.25.0",
            "huggingface_hub>=0.19.0",
            "wandb>=0.16.0",
            "safetensors>=0.4.0",
            "lm-eval",
            "python-dotenv>=1.0.0",
            "scipy>=1.11.0",
            "einops>=0.7.0",
        )
        # Copy project files to /root/project
        .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/root/project/src")
        .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/root/project/configs")
        .add_local_dir(str(PROJECT_ROOT / "mixlora"), remote_path="/root/project/mixlora")
    )
    
    @app.function(
        image=image,
        gpu=GPU_TYPE,
        volumes={VOLUME_MOUNT_PATH: volume},
        timeout=TIMEOUT_SECONDS,
        secrets=[
            modal.Secret.from_name(HF_SECRET_NAME),
            modal.Secret.from_name(WANDB_SECRET_NAME),
        ],
    )
    def train_remote(
        train_config_dict: dict = None,
        data_config_dict: dict = None,
        push_to_hub: bool = True,
    ):
        """
        Remote training function on Modal.
        
        Args:
            train_config_dict: Training configuration as dict
            data_config_dict: Data configuration as dict
            push_to_hub: Whether to push to HuggingFace Hub
            
        Returns:
            Training results
        """
        import os
        import sys
        
        # Setup paths - add project root to Python path FIRST
        sys.path.insert(0, "/root/project")
        
        # NOW we can import from project
        from configs.train_config import TrainConfig
        from configs.env_config import EnvConfig
        from configs.data_config import DataConfig
        from src.model.trainer import train_model
        
        # Create configs from dicts
        train_config = TrainConfig()
        if train_config_dict:
            for k, v in train_config_dict.items():
                if hasattr(train_config, k):
                    setattr(train_config, k, v)
        
        data_config = DataConfig()
        if data_config_dict:
            for k, v in data_config_dict.items():
                if hasattr(data_config, k):
                    setattr(data_config, k, v)
        
        env_config = EnvConfig.from_env()
        
        # Update output dir for Modal volume
        train_config.output_dir = os.path.join("/vol", "outputs")
        
        # Run training
        model = train_model(
            train_config=train_config,
            env_config=env_config,
            data_config=data_config,
            push_to_hub=push_to_hub,
        )
        
        # Commit volume
        volume.commit()
        
        return {"status": "success", "output_dir": train_config.output_dir}
    
    
    @app.function(
        image=image,
        gpu=GPU_TYPE,
        volumes={VOLUME_MOUNT_PATH: volume},
        timeout=TIMEOUT_SECONDS,
        secrets=[
            modal.Secret.from_name(HF_SECRET_NAME),
        ],
    )
    def evaluate_remote(
        tasks: list = None,
        limit: int = None,
    ):
        """
        Remote evaluation function on Modal.
        
        Args:
            tasks: List of evaluation tasks
            limit: Sample limit per task
            
        Returns:
            Evaluation results
        """
        import sys
        
        # Setup paths - add project root to Python path FIRST
        sys.path.insert(0, "/root/project")
        
        # NOW we can import from project
        from configs.env_config import EnvConfig
        from src.model.evaluator import evaluate_model
        
        if tasks is None:
            tasks = ["arc_easy"]
        
        env_config = EnvConfig.from_env()
        
        results = evaluate_model(
            env_config=env_config,
            tasks=tasks,
            limit=limit,
        )
        
        return {"status": "success", "results": results}
    
    
    @app.local_entrypoint()
    def main(
        mode: str = "train",
        num_samples: int = 17000,
        max_steps: int = -1,
        push_to_hub: bool = True,
    ):
        """
        Local entrypoint for Modal CLI.
        
        Usage:
            modal run src/modal/modal_entry.py --mode train
            modal run src/modal/modal_entry.py --mode evaluate
        """
        if mode == "train":
            train_config_dict = {
                "max_steps": max_steps,
            }
            data_config_dict = {
                "num_samples": num_samples,
            }
            result = train_remote.remote(
                train_config_dict=train_config_dict,
                data_config_dict=data_config_dict,
                push_to_hub=push_to_hub,
            )
            print(f"Training result: {result}")
            
        elif mode == "evaluate":
            result = evaluate_remote.remote(
                tasks=["arc_easy"],
                limit=10,
            )
            print(f"Evaluation result: {result}")
        
        else:
            print(f"Unknown mode: {mode}. Use 'train' or 'evaluate'.")

else:
    # Placeholder functions when Modal is not available
    def train_remote(*args, **kwargs):
        raise ImportError("Modal is not installed. Run: pip install modal")
    
    def evaluate_remote(*args, **kwargs):
        raise ImportError("Modal is not installed. Run: pip install modal")
