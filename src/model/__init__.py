"""Model module for GPT-OSS MixLoRA."""

from .load_model import (
    load_base_model,
    load_tokenizer,
    freeze_base_model,
    count_trainable_parameters,
    NormalAugmenter,
    create_mixlora_config_for_gptoss,
    inject_mixlora_into_gptoss,
    initialize_mixlora_weights_for_gptoss,
    extract_mixlora_weights_from_model,
    save_mixlora_adapter,
    verify_injection,
    print_trainable_parameters_detailed,
)
from .trainer import train_model
from .evaluator import evaluate_model, prepare_model_for_eval
from .infer import generate_text, load_model_for_inference, chat

__all__ = [
    # Load model
    "load_base_model",
    "load_tokenizer",
    "freeze_base_model",
    "count_trainable_parameters",
    # MixLoRA
    "NormalAugmenter",
    "create_mixlora_config_for_gptoss",
    "inject_mixlora_into_gptoss",
    "initialize_mixlora_weights_for_gptoss",
    "extract_mixlora_weights_from_model",
    "save_mixlora_adapter",
    "verify_injection",
    "print_trainable_parameters_detailed",
    # Training
    "train_model",
    # Evaluation
    "evaluate_model",
    "prepare_model_for_eval",
    # Inference
    "generate_text",
    "load_model_for_inference",
    "chat",
]