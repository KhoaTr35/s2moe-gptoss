"""
Training module for GPT-OSS MixLoRA.
"""
import os
import tempfile
import torch
from typing import Optional

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import HfApi, login

from configs.train_config import TrainConfig
from configs.env_config import EnvConfig
from configs.data_config import DataConfig
from src.data.dataset_utils import create_commonsense_dataset
from .load_model import (
    load_base_model,
    load_tokenizer,
    freeze_base_model,
    count_trainable_parameters,
    create_mixlora_config_for_gptoss,
    initialize_mixlora_weights_for_gptoss,
    inject_mixlora_into_gptoss,
    extract_mixlora_weights_from_model,
    save_mixlora_adapter,
    print_trainable_parameters_detailed,
)


def train_model(
    train_config: Optional[TrainConfig] = None,
    env_config: Optional[EnvConfig] = None,
    data_config: Optional[DataConfig] = None,
    push_to_hub: bool = True,
) -> PreTrainedModel:
    """
    Main training function for GPT-OSS MixLoRA.
    
    Args:
        train_config: Training configuration
        env_config: Environment configuration
        data_config: Data configuration
        push_to_hub: Whether to push to HuggingFace Hub
        
    Returns:
        Trained model
    """
    if train_config is None:
        train_config = TrainConfig()
    if env_config is None:
        env_config = EnvConfig.from_env()
    if data_config is None:
        data_config = DataConfig()
    
    print("="*60)
    print("GPT-OSS MixLoRA Training Script")
    print("="*60)
    
    # 1. Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model = load_base_model(
        model_id=train_config.model_id,
        torch_dtype=train_config.torch_dtype,
        device_map=train_config.device_map,
        trust_remote_code=train_config.trust_remote_code,
    )
    tokenizer = load_tokenizer(
        model_id=train_config.model_id,
        trust_remote_code=train_config.trust_remote_code,
    )
    
    # 2. Freeze base model
    print("\n2. Freezing base model parameters...")
    model = freeze_base_model(model)
    
    # 3. Create MixLoRA config
    print("\n3. Creating MixLoRA config...")
    mixlora_config = create_mixlora_config_for_gptoss(
        base_model=train_config.model_id,
        r=train_config.mixlora.r,
        lora_alpha=train_config.mixlora.lora_alpha,
        lora_dropout=train_config.mixlora.lora_dropout,
        num_experts=train_config.mixlora.num_experts,
        top_k=train_config.mixlora.top_k,
        target_modules=train_config.mixlora.target_modules,
        use_augmentation=train_config.mixlora.use_augmentation,
        alpha_scale=train_config.mixlora.alpha_scale,
        beta_scale=train_config.mixlora.beta_scale,
    )
    
    # 4. Initialize MixLoRA weights
    print("\n4. Initializing MixLoRA weights...")
    mixlora_weights = initialize_mixlora_weights_for_gptoss(model, mixlora_config)
    
    # 5. Inject MixLoRA into model
    print("\n5. Injecting MixLoRA into model...")
    model = inject_mixlora_into_gptoss(model, mixlora_config, mixlora_weights)
    
    # 6. Count trainable parameters
    print("\n6. Counting trainable parameters after injection...")
    count_trainable_parameters(model)
    print_trainable_parameters_detailed(model)
    
    # 7. Create dataset
    print("\n7. Creating dataset...")
    train_dataset = create_commonsense_dataset(
        tokenizer, 
        num_samples=data_config.num_samples,
        max_length=data_config.max_length,
        config=data_config,
    )
    
    # 8. Setup training arguments
    print("\n8. Setting up training arguments...")
    training_args = TrainingArguments(**train_config.to_training_arguments_dict())
    
    # 9. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 10. Initialize Trainer
    print("\n9. Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 11. Train
    print("\n10. Starting training...")
    print("="*60)
    trainer.train()
    print("="*60)
    
    print("\n✅ Training completed!")
    
    # 12. Final parameter count
    print("\n11. Final parameter count...")
    count_trainable_parameters(model)
    
    # 13. Extract trained MixLoRA weights
    print("\n12. Extracting trained MixLoRA weights...")
    trained_weights = extract_mixlora_weights_from_model(model, mixlora_config)
    
    aug_keys = [k for k in trained_weights.keys() if "augment" in k]
    print(f"Extracted {len(trained_weights)} tensors.")
    if len(aug_keys) > 0:
        print(f"✅ SUCCESS: Found {len(aug_keys)} augmentation parameters.")
    else:
        print("⚠️ WARNING: No augmentation parameters found!")

    # 14. Save Adapter
    print("\n13. Saving adapter...")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = os.path.join(tmpdir, "adapter")
        save_mixlora_adapter(model, mixlora_config, trained_weights, adapter_dir)
        
        model.config.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        
        # 15. Push to HuggingFace Hub
        if push_to_hub and env_config.hf_token:
            print("\n14. Pushing to HuggingFace Hub...")
            login(token=env_config.hf_token)
            api = HfApi()
            api.create_repo(
                repo_id=env_config.full_repo_id, 
                token=env_config.hf_token, 
                exist_ok=True, 
                repo_type="model"
            )
            api.upload_folder(
                folder_path=adapter_dir,
                repo_id=env_config.full_repo_id,
                token=env_config.hf_token,
                commit_message="Upload trained MixLoRA adapter for gpt-oss-20b",
            )
            print(f"✅ Model uploaded to: https://huggingface.co/{env_config.full_repo_id}")
        else:
            # Save locally
            local_adapter_dir = os.path.join(train_config.output_dir, "adapter")
            os.makedirs(local_adapter_dir, exist_ok=True)
            save_mixlora_adapter(model, mixlora_config, trained_weights, local_adapter_dir)
            model.config.save_pretrained(local_adapter_dir)
            tokenizer.save_pretrained(local_adapter_dir)
            print(f"✅ Adapter saved locally to: {local_adapter_dir}")
    
    return model
