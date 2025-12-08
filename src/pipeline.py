"""
Main pipeline orchestrator for GPT-OSS MixLoRA.
Sequences: Training -> Evaluation -> Upload
"""
from typing import Optional

from configs.train_config import TrainConfig
from configs.env_config import EnvConfig
from configs.data_config import DataConfig
from src.model.trainer import train_model
from src.model.evaluator import evaluate_model
from src.model.infer import generate_text, load_model_for_inference
from src.utils.timer import Timer
from src.utils.logging import setup_logging, setup_wandb


class Pipeline:
    """
    Main pipeline class for orchestrating the full training and evaluation workflow.
    
    Usage:
        pipeline = Pipeline()
        pipeline.run_full()
        
        # Or run individual steps:
        pipeline.train()
        pipeline.evaluate()
        pipeline.inference("Hello world")
    """
    
    def __init__(
        self,
        train_config: Optional[TrainConfig] = None,
        env_config: Optional[EnvConfig] = None,
        data_config: Optional[DataConfig] = None,
    ):
        """
        Initialize pipeline with configurations.
        
        Args:
            train_config: Training configuration
            env_config: Environment configuration
            data_config: Data configuration
        """
        self.train_config = train_config or TrainConfig()
        self.env_config = env_config or EnvConfig.from_env()
        self.data_config = data_config or DataConfig()
        
        self.model = None
        self.tokenizer = None
        self.logger = setup_logging()
    
    def setup(self):
        """Setup environment and logging."""
        self.logger.info("Setting up pipeline...")
        
        # Setup environment variables
        self.env_config.setup_environment()
        
        # Validate configuration
        if not self.env_config.validate():
            self.logger.warning("Environment validation failed. Some features may not work.")
        
        # Setup W&B if enabled
        if self.train_config.report_to == "wandb":
            setup_wandb(
                project=self.env_config.wandb_project,
                entity=self.env_config.wandb_entity,
                config={
                    "model_id": self.train_config.model_id,
                    "mixlora_r": self.train_config.mixlora.r,
                    "num_experts": self.train_config.mixlora.num_experts,
                    "top_k": self.train_config.mixlora.top_k,
                    "learning_rate": self.train_config.learning_rate,
                }
            )
        
        self.logger.info("Pipeline setup complete")
    
    def train(self, push_to_hub: bool = True):
        """
        Run training step.
        
        Args:
            push_to_hub: Whether to push to HuggingFace Hub
            
        Returns:
            Trained model
        """
        self.logger.info("Starting training...")
        
        with Timer("Training"):
            self.model = train_model(
                train_config=self.train_config,
                env_config=self.env_config,
                data_config=self.data_config,
                push_to_hub=push_to_hub,
            )
        
        self.logger.info("Training complete")
        return self.model
    
    def evaluate(
        self,
        tasks: list = None,
        limit: Optional[int] = None,
    ):
        """
        Run evaluation step.
        
        Args:
            tasks: List of evaluation tasks
            limit: Sample limit per task
            
        Returns:
            Evaluation results
        """
        if tasks is None:
            tasks = ["arc_easy", "arc_challenge", "hellaswag"]
        
        self.logger.info(f"Starting evaluation on tasks: {tasks}")
        
        with Timer("Evaluation"):
            results = evaluate_model(
                model_id=self.train_config.model_id,
                env_config=self.env_config,
                tasks=tasks,
                limit=limit,
            )
        
        self.logger.info("Evaluation complete")
        return results
    
    def inference(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        **generation_kwargs,
    ):
        """
        Run inference on the trained model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.logger.info("Loading model for inference...")
            self.model, self.tokenizer = load_model_for_inference(
                model_id=self.train_config.model_id,
                env_config=self.env_config,
            )
        
        self.logger.info(f"Generating response for: {prompt[:50]}...")
        
        response = generate_text(
            prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
        
        return response
    
    def run_full(
        self,
        push_to_hub: bool = True,
        run_eval: bool = True,
        eval_tasks: list = None,
        eval_limit: Optional[int] = None,
    ):
        """
        Run the full pipeline: setup -> train -> evaluate.
        
        Args:
            push_to_hub: Whether to push to HuggingFace Hub
            run_eval: Whether to run evaluation
            eval_tasks: Evaluation tasks
            eval_limit: Sample limit for evaluation
            
        Returns:
            Dict with training and evaluation results
        """
        results = {}
        
        with Timer("Full Pipeline"):
            # Setup
            self.setup()
            
            # Train
            self.train(push_to_hub=push_to_hub)
            results["training"] = "completed"
            
            # Evaluate
            if run_eval:
                eval_results = self.evaluate(
                    tasks=eval_tasks,
                    limit=eval_limit,
                )
                results["evaluation"] = eval_results
        
        self.logger.info("Full pipeline complete")
        return results
    
    def cleanup(self):
        """Cleanup resources."""
        import gc
        
        self.model = None
        self.tokenizer = None
        
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.logger.info("Cleanup complete")


def run_pipeline(
    mode: str = "full",
    push_to_hub: bool = True,
    num_samples: int = 17000,
    max_steps: int = -1,
    eval_tasks: list = None,
    eval_limit: Optional[int] = None,
):
    """
    Convenience function to run pipeline.
    
    Args:
        mode: Pipeline mode ('full', 'train', 'evaluate')
        push_to_hub: Whether to push to Hub
        num_samples: Number of training samples
        max_steps: Maximum training steps
        eval_tasks: Evaluation tasks
        eval_limit: Evaluation sample limit
        
    Returns:
        Pipeline results
    """
    # Create configs
    train_config = TrainConfig()
    train_config.max_steps = max_steps
    
    data_config = DataConfig()
    data_config.num_samples = num_samples
    
    env_config = EnvConfig.from_env()
    
    # Create and run pipeline
    pipeline = Pipeline(
        train_config=train_config,
        env_config=env_config,
        data_config=data_config,
    )
    
    if mode == "full":
        return pipeline.run_full(
            push_to_hub=push_to_hub,
            run_eval=True,
            eval_tasks=eval_tasks,
            eval_limit=eval_limit,
        )
    elif mode == "train":
        return pipeline.train(push_to_hub=push_to_hub)
    elif mode == "evaluate":
        return pipeline.evaluate(tasks=eval_tasks, limit=eval_limit)
    else:
        raise ValueError(f"Unknown mode: {mode}")
