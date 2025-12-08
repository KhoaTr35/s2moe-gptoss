#!/usr/bin/env python3
"""
Main entry point for GPT-OSS MixLoRA project.

Usage:
    python run.py train [--num_samples 17000] [--max_steps -1] [--push_to_hub]
    python run.py evaluate [--tasks arc_easy,hellaswag] [--limit 10]
    python run.py infer --text "Hello, world"
    python run.py full [--push_to_hub] [--no_eval]
    
Examples:
    # Full training with 17k samples
    python run.py train --num_samples 17000 --push_to_hub
    
    # Quick test run
    python run.py train --num_samples 1000 --max_steps 10
    
    # Evaluate on specific tasks
    python run.py evaluate --tasks arc_easy,arc_challenge --limit 100
    
    # Run inference
    python run.py infer --text "What is the capital of France?"
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.train_config import TrainConfig
from configs.env_config import EnvConfig
from configs.data_config import DataConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPT-OSS MixLoRA Training and Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--num_samples", type=int, default=17000, help="Number of training samples")
    train_parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    train_parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    train_parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    train_parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    train_parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    train_parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    train_parser.add_argument("--num_experts", type=int, default=4, help="Number of MoE experts")
    train_parser.add_argument("--top_k", type=int, default=2, help="Top-k experts per token")
    train_parser.add_argument("--use_augmentation", action="store_true", default=True, help="Use dual-path augmentation")
    train_parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--tasks", type=str, default="arc_easy", help="Comma-separated list of tasks")
    eval_parser.add_argument("--limit", type=int, default=None, help="Limit samples per task")
    eval_parser.add_argument("--adapter_repo", type=str, default=None, help="HuggingFace adapter repo ID")
    eval_parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    eval_parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--text", type=str, required=True, help="Input text/prompt")
    infer_parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens")
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    infer_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    infer_parser.add_argument("--adapter_repo", type=str, default=None, help="HuggingFace adapter repo ID")
    
    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full pipeline (train + evaluate)")
    full_parser.add_argument("--num_samples", type=int, default=17000, help="Number of training samples")
    full_parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps")
    full_parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    full_parser.add_argument("--no_eval", action="store_true", help="Skip evaluation")
    full_parser.add_argument("--eval_tasks", type=str, default="arc_easy", help="Evaluation tasks")
    full_parser.add_argument("--eval_limit", type=int, default=None, help="Evaluation sample limit")
    
    return parser.parse_args()


def cmd_train(args):
    """Handle train command."""
    from src.model.trainer import train_model
    
    # Create configs
    train_config = TrainConfig()
    train_config.max_steps = args.max_steps
    train_config.per_device_train_batch_size = args.batch_size
    train_config.learning_rate = args.learning_rate
    train_config.output_dir = args.output_dir
    train_config.mixlora.r = args.lora_r
    train_config.mixlora.num_experts = args.num_experts
    train_config.mixlora.top_k = args.top_k
    train_config.mixlora.use_augmentation = args.use_augmentation
    
    if args.no_wandb:
        train_config.report_to = "none"
    
    data_config = DataConfig()
    data_config.num_samples = args.num_samples
    data_config.max_length = args.max_length
    
    env_config = EnvConfig.from_env()
    
    print("="*60)
    print("GPT-OSS MixLoRA Training")
    print("="*60)
    print(f"  Samples: {args.num_samples}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  LoRA Rank: {args.lora_r}")
    print(f"  Experts: {args.num_experts}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Push to Hub: {args.push_to_hub}")
    print("="*60)
    
    model = train_model(
        train_config=train_config,
        env_config=env_config,
        data_config=data_config,
        push_to_hub=args.push_to_hub,
    )
    
    print("\n✅ Training complete!")
    return model


def cmd_evaluate(args):
    """Handle evaluate command."""
    from src.model.evaluator import evaluate_model
    
    tasks = args.tasks.split(",")
    env_config = EnvConfig.from_env()
    
    if args.adapter_repo:
        env_config.hf_repo_name = args.adapter_repo.split("/")[-1]
        if "/" in args.adapter_repo:
            env_config.hf_username = args.adapter_repo.split("/")[0]
    
    print("="*60)
    print("GPT-OSS MixLoRA Evaluation")
    print("="*60)
    print(f"  Tasks: {tasks}")
    print(f"  Limit: {args.limit}")
    print(f"  Adapter: {env_config.full_repo_id}")
    print("="*60)
    
    results = evaluate_model(
        env_config=env_config,
        tasks=tasks,
        limit=args.limit,
    )
    
    print("\n✅ Evaluation complete!")
    print(results)
    return results


def cmd_infer(args):
    """Handle inference command."""
    from src.model.infer import generate_text, load_model_for_inference
    
    env_config = EnvConfig.from_env()
    
    if args.adapter_repo:
        env_config.hf_repo_name = args.adapter_repo.split("/")[-1]
        if "/" in args.adapter_repo:
            env_config.hf_username = args.adapter_repo.split("/")[0]
    
    print("="*60)
    print("GPT-OSS MixLoRA Inference")
    print("="*60)
    print(f"  Input: {args.text[:50]}...")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print("="*60)
    
    model, tokenizer = load_model_for_inference(env_config=env_config)
    
    response = generate_text(
        prompt=args.text,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    print("\n" + "="*60)
    print("Response:")
    print("="*60)
    print(response)
    print("="*60)
    
    return response


def cmd_full(args):
    """Handle full pipeline command."""
    from src.pipeline import Pipeline
    
    train_config = TrainConfig()
    train_config.max_steps = args.max_steps
    
    data_config = DataConfig()
    data_config.num_samples = args.num_samples
    
    env_config = EnvConfig.from_env()
    
    eval_tasks = args.eval_tasks.split(",") if args.eval_tasks else None
    
    print("="*60)
    print("GPT-OSS MixLoRA Full Pipeline")
    print("="*60)
    print(f"  Samples: {args.num_samples}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Push to Hub: {args.push_to_hub}")
    print(f"  Run Eval: {not args.no_eval}")
    print(f"  Eval Tasks: {eval_tasks}")
    print("="*60)
    
    pipeline = Pipeline(
        train_config=train_config,
        env_config=env_config,
        data_config=data_config,
    )
    
    results = pipeline.run_full(
        push_to_hub=args.push_to_hub,
        run_eval=not args.no_eval,
        eval_tasks=eval_tasks,
        eval_limit=args.eval_limit,
    )
    
    print("\n✅ Full pipeline complete!")
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command is None:
        print("No command specified. Use --help for usage information.")
        print("\nAvailable commands: train, evaluate, infer, full")
        return 1
    
    # Dispatch to command handler
    handlers = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
        "full": cmd_full,
    }
    
    handler = handlers.get(args.command)
    if handler:
        try:
            handler(args)
            return 0
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return 130
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
