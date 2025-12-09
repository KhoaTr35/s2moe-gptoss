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
        adapter_repo: str = None,
    ):
        """
        Remote evaluation function on Modal.
        
        Args:
            tasks: List of evaluation tasks
            limit: Sample limit per task
            adapter_repo: HuggingFace adapter repo to load
            
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
            adapter_repo_id=adapter_repo,
        )
        
        return {"status": "success", "results": results}
    
    
    @app.function(
        image=image,
        gpu=GPU_TYPE,
        volumes={VOLUME_MOUNT_PATH: volume},
        timeout=TIMEOUT_SECONDS,
        secrets=[
            modal.Secret.from_name(HF_SECRET_NAME),
        ],
    )
    def infer_remote(
        text: str,
        adapter_repo: str = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_samples: int = 1,
    ):
        """
        Remote inference function on Modal.
        
        Args:
            text: Input prompt
            adapter_repo: HuggingFace adapter repo to load
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            num_samples: Number of responses to generate
            
        Returns:
            Generated text(s)
        """
        import sys
        import torch
        
        # Setup paths
        sys.path.insert(0, "/root/project")
        
        from configs.env_config import EnvConfig
        from src.model.infer import generate_text, load_model_for_inference
        
        env_config = EnvConfig.from_env()
        
        # Use provided adapter repo or default from env
        if adapter_repo:
            adapter_repo_id = adapter_repo
        else:
            adapter_repo_id = env_config.full_repo_id
        
        print(f"Loading model with adapter: {adapter_repo_id}")
        
        # Load model
        model, tokenizer = load_model_for_inference(
            model_id="openai/gpt-oss-20b",
            adapter_repo_id=adapter_repo_id,
            dtype=torch.bfloat16,
        )
        
        # Generate
        print(f"Generating response for: {text[:50]}...")
        response = generate_text(
            prompt=text,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
        )
        
        return {"status": "success", "prompt": text, "response": response}
    
    
    @app.local_entrypoint()
    def main(
        mode: str = "train",
        num_samples: int = 17000,
        max_steps: int = -1,
        push_to_hub: bool = True,
        eval_tasks: str = "arc_easy", # use underscore for modal CLI
        eval_limit: int = None,
        adapter_repo: str = "twanghcmut/mixlora-gpt-oss-experimental-run",
        text: str = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ):
        """
        Local entrypoint for Modal CLI.
        
        Usage:
            modal run src/modal/modal_entry.py --mode train
            modal run src/modal/modal_entry.py --mode evaluate --eval-tasks arc_easy
            modal run src/modal/modal_entry.py --mode evaluate --eval-tasks arc_easy,hellaswag --eval-limit 100
            modal run src/modal/modal_entry.py --mode infer --text "What is AI?"
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
            # Parse tasks string to list
            task_list = [t.strip() for t in eval_tasks.split(",")]
            
            result = evaluate_remote.remote(
                tasks=task_list,
                limit=eval_limit,
                adapter_repo=adapter_repo,
            )
            print(f"Evaluation result: {result}")
        
        elif mode == "infer":
            if text is None:
                print("ERROR: --text is required for inference mode")
                return
            
            result = infer_remote.remote(
                text=text,
                adapter_repo=adapter_repo,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            print(f"\n{'='*60}")
            print(f"Prompt: {result['prompt']}")
            print(f"{'='*60}")
            print(f"Response:\n{result['response']}")
            print(f"{'='*60}")
        
        else:
            print(f"Unknown mode: {mode}. Use 'train', 'evaluate', or 'infer'.")

else:
    # Placeholder functions when Modal is not available
    def train_remote(*args, **kwargs):
        raise ImportError("Modal is not installed. Run: pip install modal")
    
    def evaluate_remote(*args, **kwargs):
        raise ImportError("Modal is not installed. Run: pip install modal")
    
    def infer_remote(*args, **kwargs):
        raise ImportError("Modal is not installed. Run: pip install modal")
