"""
Inference module for GPT-OSS MixLoRA.
"""
import torch
from typing import Optional, List, Union

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from mixlora import load_adapter_weights
from configs.env_config import EnvConfig
from .load_model import inject_mixlora_into_gptoss


def load_model_for_inference(
    model_id: str = "openai/gpt-oss-20b",
    adapter_repo_id: Optional[str] = None,
    env_config: Optional[EnvConfig] = None,
    device: str = "cuda",
    target_gpu: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load model with MixLoRA adapter for inference.
    
    Args:
        model_id: Base model ID
        adapter_repo_id: Adapter repo ID
        env_config: Environment configuration
        device: Device to use
        target_gpu: Target GPU
        dtype: Data type
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if env_config is None:
        env_config = EnvConfig.from_env()
    
    if adapter_repo_id is None:
        adapter_repo_id = env_config.full_repo_id
    
    print(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=target_gpu,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading adapter from: {adapter_repo_id}")
    loaded_config, loaded_weights = load_adapter_weights(
        adapter_repo_id,
        adapter_name="default",
        device=device,
        dtype=dtype,
    )
    
    print("Injecting adapter...")
    model = inject_mixlora_into_gptoss(model, loaded_config, loaded_weights)
    model.eval()
    
    print("✅ Model ready for inference")
    return model, tokenizer


def generate_text(
    prompt: Union[str, List[str]],
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model_id: str = "openai/gpt-oss-20b",
    adapter_repo_id: Optional[str] = None,
    max_length: int = 256,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    do_sample: bool = True,
    num_return_sequences: int = 1,
) -> Union[str, List[str]]:
    """
    Generate text using the MixLoRA model.
    
    Args:
        prompt: Input prompt(s)
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        model_id: Base model ID (if loading)
        adapter_repo_id: Adapter repo ID (if loading)
        max_length: Maximum total length
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
        repetition_penalty: Repetition penalty
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to return
        
    Returns:
        Generated text(s)
    """
    # Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model_for_inference(
            model_id=model_id,
            adapter_repo_id=adapter_repo_id,
        )
    
    # Handle single prompt
    is_single = isinstance(prompt, str)
    if is_single:
        prompt = [prompt]
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        generation_kwargs = {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = max_new_tokens
        else:
            generation_kwargs["max_length"] = max_length
        
        outputs = model.generate(**inputs, **generation_kwargs)
    
    # Decode
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    if is_single and num_return_sequences == 1:
        return generated_texts[0]
    
    return generated_texts


def format_instruction_prompt(
    instruction: str,
    input_text: str = "",
) -> str:
    """
    Format a prompt in Alpaca instruction format.
    
    Args:
        instruction: The instruction/question
        input_text: Optional additional input
        
    Returns:
        Formatted prompt
    """
    if input_text and len(input_text.strip()) > 0:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def chat(
    instruction: str,
    input_text: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **generation_kwargs,
) -> str:
    """
    Simple chat interface for instruction-following.
    
    Args:
        instruction: The instruction/question
        input_text: Optional additional input
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Model response
    """
    prompt = format_instruction_prompt(instruction, input_text)
    
    response = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        **generation_kwargs,
    )
    
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response
