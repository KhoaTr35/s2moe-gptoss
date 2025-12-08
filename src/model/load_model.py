"""
Model loading utilities for GPT-OSS MixLoRA.
Handles base model loading, tokenizer setup, LoRA injection, and weight management.
"""
import os
import json
import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from mixlora import MixLoraConfig, load_adapter_weights
from mixlora.lora_linear import LoraLinear
from mixlora.model import MixLoraSparseMoe, _slice_tensor, _compatible_model_types
from mixlora.config import ACT2FN

from configs.train_config import TrainConfig


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_base_model(
    model_id: str = "openai/gpt-oss-20b",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> PreTrainedModel:
    """
    Load base GPT-OSS model from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID
        torch_dtype: Model dtype (bfloat16, float16, float32)
        device_map: Device placement strategy
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Loaded model
    """
    print(f"Loading base model: {model_id}")
    print(f"  dtype: {torch_dtype}")
    print(f"  device_map: {device_map}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    
    print(f"✅ Model loaded successfully")
    return model


def load_tokenizer(
    model_id: str = "openai/gpt-oss-20b",
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer:
    """
    Load tokenizer for GPT-OSS model.
    
    Args:
        model_id: HuggingFace model ID
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Loaded tokenizer with pad_token set
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def freeze_base_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Freeze all base model parameters for efficient fine-tuning.
    
    Args:
        model: The model to freeze
        
    Returns:
        Model with frozen parameters
    """
    for param in model.parameters():
        param.requires_grad = False
    
    print("✅ Base model parameters frozen")
    return model


def count_trainable_parameters(model: PreTrainedModel) -> Tuple[int, int]:
    """
    Count trainable and total parameters.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    return trainable_params, total_params


# ============================================================================
# AUGMENTATION COMPONENTS
# ============================================================================

class NormalAugmenter(nn.Module):
    """
    Trainable Augmenter using nn.InstanceNorm1d.
    - Learns optimal feature statistics (affine=True).
    - Learns optimal noise scaling (alpha/beta as parameters).
    - Auto-Device check & SeqLen check (Inference Safety).
    """
    def __init__(self, feature_size, alpha_scale=1.0, beta_scale=1.0, eps=1e-6, *args, **kwargs):
        super().__init__()
        
        # Trainable Instance Norm (affine=True -> Learnable Gamma/Beta)
        self.instance_norm = nn.InstanceNorm1d(feature_size, eps=eps, affine=True)
        
        # Trainable Noise Scaling
        self.alpha_scale = nn.Parameter(torch.tensor(float(alpha_scale)))
        self.beta_scale = nn.Parameter(torch.tensor(float(beta_scale)))

    def forward(self, features, *args, **kwargs):
        """
        Args:
            features: Tensor of shape [Batch, Seq, Hidden] or [Batch*Seq, Hidden]
        """
        # Auto-Device Placement
        target_device = features.device
        if self.alpha_scale.device != target_device:
            self.to(target_device)

        input_dtype = features.dtype
        
        # Shape Handling
        is_flat = features.dim() == 2
        if is_flat:
            total_tokens, hidden_dim = features.shape
            x = features.view(1, total_tokens, hidden_dim)
        else:
            x = features

        x_permuted = x.permute(0, 2, 1)  # [Batch, Hidden, Seq]

        # SAFETY CHECK: Skip if Sequence Length is <= 1
        if x_permuted.size(2) <= 1:
            return features

        # Apply Norm & Noise (FP32 for stability)
        x_permuted_f32 = x_permuted.to(torch.float32)
        x_norm = self.instance_norm(x_permuted_f32)
        
        alpha_noise = torch.randn_like(x_norm) * self.alpha_scale
        beta_noise = torch.randn_like(x_norm) * self.beta_scale
        
        augmented_norm = (1 + alpha_noise) * x_norm + beta_noise

        # Restore
        augmented = augmented_norm.to(input_dtype)
        augmented = augmented.permute(0, 2, 1)
        
        if is_flat:
            augmented = augmented.view(-1, hidden_dim)
            
        return augmented

    @classmethod
    def code(cls) -> str:
        return 'normal_gaussian_trainable'


# ============================================================================
# MIXLORA CONFIG CREATION
# ============================================================================

def create_mixlora_config_for_gptoss(
    base_model: str = "openai/gpt-oss-20b",
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    num_experts: int = 4,
    top_k: int = 2,
    target_modules: Optional[Dict[str, bool]] = None,
    use_augmentation: bool = True,
    alpha_scale: float = 1.0,
    beta_scale: float = 1.0,
) -> MixLoraConfig:
    """
    Create MixLoRA config for GPT-OSS model with Dual-Path Augmentation support.
    
    Args:
        base_model: Base model ID
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout rate
        num_experts: Number of MoE experts
        top_k: Number of experts to route to
        target_modules: Dict of module names to apply LoRA to
        use_augmentation: Enable Dual-Path Augmentation
        alpha_scale: Multiplicative noise intensity
        beta_scale: Additive noise intensity
        
    Returns:
        Configured MixLoraConfig
    """
    if target_modules is None:
        target_modules = {
            "q_proj": True,
            "k_proj": True,
            "v_proj": True,
            "o_proj": True,
            "gate_proj": True,
            "up_proj": True,
            "down_proj": True,
        }
    
    config_dict = {
        "peft_type": "MIXLORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "routing_strategy": "mixlora",
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": [k for k, v in target_modules.items() if v],
        "num_experts": num_experts,
        "top_k": top_k,
        "router_init_range": 0.02,
        "jitter_noise": 0.0,
        "router_loss": True,
        "router_aux_loss_coef": 0.001,
    }
    
    config = MixLoraConfig.from_config(config_dict)
    
    # Augmentation fields
    config.use_augmentation = use_augmentation
    config.alpha_scale = alpha_scale
    config.beta_scale = beta_scale
    
    # Other fixed configurations
    config.adapter_name_ = "default"
    config.dtype_ = torch.float32
    config.act_fn_ = "silu"
    
    return config


# ============================================================================
# GPT-OSS FORWARD FUNCTION
# ============================================================================

def _gptoss_forward(
    self,
    expert_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    input_dtype: torch.dtype,
):
    """
    GPT-OSS MLP forward with MixLoRA experts.
    
    Architecture specifics:
    - Weights shape: [num_experts, in_features, out_features]
    - Activation: SiLU (SwiGLU structure)
    - Empty Experts: Returns (0, hidden) tensor to prevent propagation errors.
    """
    base_experts = self.base_layer_.experts
    act_fn = self.act_fn_ if hasattr(self, 'act_fn_') and self.act_fn_ is not None else torch.nn.functional.silu

    # Get Projections
    if hasattr(base_experts, 'gate_up_proj'):
        gate_up_proj = base_experts.gate_up_proj
        gate_up_bias = getattr(base_experts, 'gate_up_proj_bias', None)
    else:
        raise ValueError("Critical: 'gate_up_proj' not found in GPT-OSS experts.")
        
    down_proj = base_experts.down_proj
    down_bias = getattr(base_experts, 'down_proj_bias', None)
    
    final_expert_states = []
    
    for expert_idx in range(self.num_experts_):
        _, top_x = torch.where(expert_mask[expert_idx])
        
        # EMPTY EXPERT HANDLING
        if top_x.numel() == 0:
            empty_tensor = torch.empty(
                (0, hidden_states.shape[-1]), 
                dtype=input_dtype, 
                device=hidden_states.device
            )
            final_expert_states.append(empty_tensor)
            continue
        
        expert_hidden = _slice_tensor(hidden_states, top_x, input_dtype)
        
        # Gate-Up Projection
        if isinstance(gate_up_proj, torch.nn.Linear):
            gate_up_output = gate_up_proj(expert_hidden)
        else:
            gate_up_weight = gate_up_proj[expert_idx] 
            gate_up_output = torch.matmul(expert_hidden, gate_up_weight)
            
            if gate_up_bias is not None:
                gate_up_output = gate_up_output + gate_up_bias[expert_idx]
        
        # Split Gate/Up
        intermediate_size = gate_up_output.shape[-1] // 2
        gate_output = gate_up_output[..., :intermediate_size]
        up_output = gate_up_output[..., intermediate_size:]
        
        # Apply LoRA (Gate)
        lora_gate = self.experts_.get(f"experts.{expert_idx}.gate_proj", None)
        if lora_gate is not None:
            gate_output = lora_gate.lora_forward(gate_output, expert_hidden)
        
        # Apply LoRA (Up)
        lora_up = self.experts_.get(f"experts.{expert_idx}.up_proj", None)
        if lora_up is not None:
            up_output = lora_up.lora_forward(up_output, expert_hidden)
        
        # Activation (SwiGLU)
        act_result = act_fn(gate_output) * up_output
        
        # Down Projection
        if isinstance(down_proj, torch.nn.Linear):
            down_output = down_proj(act_result)
        else:
            down_weight = down_proj[expert_idx]
            down_output = torch.matmul(act_result, down_weight)
            
            if down_bias is not None:
                down_output = down_output + down_bias[expert_idx]
        
        # Apply LoRA (Down)
        lora_down = self.experts_.get(f"experts.{expert_idx}.down_proj", None)
        if lora_down is not None:
            down_output = lora_down.lora_forward(down_output, act_result)
        
        final_expert_states.append(down_output)
    
    return final_expert_states


# Register GPT-OSS forward
MixLoraSparseMoe._gptoss_forward = _gptoss_forward
_compatible_model_types["gpt_oss"] = "_gptoss_forward"
_compatible_model_types["gpt-oss"] = "_gptoss_forward"
_compatible_model_types["gpt_oss_20b"] = "_gptoss_forward"


# ============================================================================
# INJECTION FUNCTIONS
# ============================================================================

def _inject_gptoss_attn_module(
    layer_idx: int,
    self_attn: torch.nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    """Inject LoRA into GPT-OSS attention."""
    print(f"  Injecting attention LoRA for layer {layer_idx}")
    
    attn_projs = []
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "out"]:
        if hasattr(self_attn, proj_name):
            attn_projs.append(proj_name)
    
    print(f"    Found projections: {attn_projs}")
    
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if proj_name not in config.target_modules_ or not config.target_modules_[proj_name]:
            continue
        if not hasattr(self_attn, proj_name):
            continue
            
        base_layer = getattr(self_attn, proj_name)
        layer_prefix_name = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"
        
        if f"{layer_prefix_name}.lora_A.weight" in weights:
            print(f"    ✓ Injecting {proj_name}")
            setattr(
                self_attn,
                proj_name,
                LoraLinear(
                    base_layer,
                    config,
                    (
                        weights[f"{layer_prefix_name}.lora_A.weight"],
                        weights[f"{layer_prefix_name}.lora_B.weight"],
                    ),
                ),
            )
        else:
            print(f"    ⚠️  No weights found for {proj_name}")


def _inject_gptoss_mlp_module(
    layer_idx: int,
    mlp: torch.nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    """Inject MixLoRA MoE into GPT-OSS MLP with PROPER WEIGHT LOADING."""
    
    if hasattr(mlp, 'mixlora_moes') and config.adapter_name_ in mlp.mixlora_moes:
        return
    
    experts = mlp.experts
    
    hidden_size = config.hidden_size_ if hasattr(config, 'hidden_size_') else 2880
    if hasattr(mlp, 'norm') and hasattr(mlp.norm, 'num_features'):
        hidden_size = mlp.norm.num_features
    
    if not hasattr(config, 'use_augmentation'):
        config.use_augmentation = True
    
    moe_layer = MixLoraSparseMoe(mlp, config)
    
    # AUGMENTATION & SMART INIT
    if moe_layer.use_augmentation_:
        print(f" 🔹 Layer {layer_idx}: Initializing Trainable Augmenter & Smart Gating")
        
        moe_layer.augmentator_ = NormalAugmenter(
            hidden_size,
            alpha_scale=getattr(config, 'alpha_scale', 0.1),
            beta_scale=getattr(config, 'beta_scale', 0.1)
        )
        
        aug_prefix = f"mixlora.layers.{layer_idx}.mlp.augmentator"
        aug_keys = [k for k in weights.keys() if k.startswith(aug_prefix)]
        if aug_keys:
            print(f"    ✓ Loading {len(aug_keys)} saved augmentator parameters")
            aug_state = {}
            for key in aug_keys:
                param_name = key.replace(f"{aug_prefix}.", "")
                aug_state[param_name] = weights[key].to(config.dtype_)
            moe_layer.augmentator_.load_state_dict(aug_state, strict=False)
        
        gate_linear1 = nn.Linear(hidden_size, hidden_size // 4)
        gate_linear2 = nn.Linear(hidden_size // 4, 1)
        
        gating_prefix = f"mixlora.layers.{layer_idx}.mlp.augment_gating"
        gating_keys = [k for k in weights.keys() if k.startswith(gating_prefix)]
        
        if gating_keys:
            print(f"    ✓ Loading {len(gating_keys)} saved gating parameters")
            if f"{gating_prefix}.0.weight" in weights:
                gate_linear1.weight.data = weights[f"{gating_prefix}.0.weight"].to(config.dtype_)
            if f"{gating_prefix}.0.bias" in weights:
                gate_linear1.bias.data = weights[f"{gating_prefix}.0.bias"].to(config.dtype_)
            if f"{gating_prefix}.2.weight" in weights:
                gate_linear2.weight.data = weights[f"{gating_prefix}.2.weight"].to(config.dtype_)
            if f"{gating_prefix}.2.bias" in weights:
                gate_linear2.bias.data = weights[f"{gating_prefix}.2.bias"].to(config.dtype_)
        else:
            print(f"    ⚙️  Using smart initialization (no saved weights)")
            nn.init.constant_(gate_linear2.bias, 5.0)
            nn.init.xavier_normal_(gate_linear2.weight, gain=0.01)
        
        moe_layer.gating_network_ = nn.Sequential(
            gate_linear1,
            nn.ReLU(),
            gate_linear2
        ).to(config.dtype_)
    
    # Act Fn Fix
    act_fn_name = getattr(config, 'act_fn_', getattr(config, 'act_fn', 'silu'))
    if isinstance(act_fn_name, str):
        act_obj = ACT2FN.get(act_fn_name, torch.nn.SiLU)
        moe_layer.act_fn_ = act_obj() if isinstance(act_obj, type) else act_obj
    else:
        moe_layer.act_fn_ = act_fn_name
    
    # Router Gate Init
    gate_key = f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"
    if gate_key in weights:
        gate_weight = weights[gate_key].to(config.dtype_)
        if gate_weight.shape == (config.num_experts_, hidden_size):
            moe_layer.gate_ = torch.nn.Parameter(gate_weight)
        elif gate_weight.shape == (hidden_size, config.num_experts_):
            moe_layer.gate_ = torch.nn.Parameter(gate_weight.T)
        else:
            print(f"    ⚠️  Gate shape mismatch, using random init")
            moe_layer.gate_ = torch.nn.Parameter(
                torch.randn(config.num_experts_, hidden_size) * config.router_init_range_
            ).to(config.dtype_)
    else:
        print(f"    ⚙️  Using random gate initialization")
        moe_layer.gate_ = torch.nn.Parameter(
            torch.randn(config.num_experts_, hidden_size) * config.router_init_range_
        ).to(config.dtype_)
    
    if not hasattr(mlp, "mixlora_moes"):
        mlp.mixlora_moes = torch.nn.ModuleDict()
    mlp.mixlora_moes[config.adapter_name_] = moe_layer
    
    if not hasattr(mlp, '_original_forward'):
        mlp._original_forward = mlp.forward
    
    def gptoss_mixlora_forward_wrapper(hidden_states, **kwargs):
        return moe_layer.forward(hidden_states)
    
    mlp.forward = gptoss_mixlora_forward_wrapper
    
    # INJECT LORA INTO EXPERTS
    if not hasattr(mlp, '_lora_expert_wrappers'):
        mlp._lora_expert_wrappers = {}
    
    experts_injected = 0
    proj_mapping = {
        'gate_proj': ('gate_up_proj', 'gate_up_proj_bias', 0),
        'up_proj': ('gate_up_proj', 'gate_up_proj_bias', 1),
        'down_proj': ('down_proj', 'down_proj_bias', None),
    }
    
    for proj_name, (weight_attr, bias_attr, split_idx) in proj_mapping.items():
        if proj_name not in config.target_modules_ or not config.target_modules_[proj_name]:
            continue
        
        if not hasattr(experts, weight_attr):
            continue
        
        expert_weight_or_module = getattr(experts, weight_attr)
        if isinstance(expert_weight_or_module, torch.nn.Linear):
            expert_weight = expert_weight_or_module.weight
        else:
            expert_weight = expert_weight_or_module
        
        target_dtype = expert_weight.data.dtype
        
        for expert_idx in range(config.num_experts_):
            layer_prefix_name = f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
            lora_a_key = f"{layer_prefix_name}.lora_A.weight"
            lora_b_key = f"{layer_prefix_name}.lora_B.weight"
            
            if lora_a_key in weights and lora_b_key in weights:
                if proj_name == 'down_proj':
                    expert_weight_slice = expert_weight[expert_idx]
                    in_features = expert_weight_slice.shape[0]
                    out_features = expert_weight_slice.shape[1]
                    base_layer = torch.nn.Linear(in_features, out_features, bias=True)
                    base_layer.weight.data = expert_weight_slice.T.clone().to(target_dtype)
                    if hasattr(experts, bias_attr):
                        expert_bias = getattr(experts, bias_attr)
                        base_layer.bias.data = expert_bias[expert_idx].clone().to(target_dtype)
                else:
                    expert_weight_slice = expert_weight[expert_idx]
                    in_features = expert_weight_slice.shape[0]
                    out_features_total = expert_weight_slice.shape[1]
                    out_features = out_features_total // 2
                    base_layer = torch.nn.Linear(in_features, out_features, bias=True)
                    
                    if split_idx == 0:
                        weight_slice = expert_weight_slice[:, :out_features]
                        base_layer.weight.data = weight_slice.T.clone().to(target_dtype)
                        if hasattr(experts, bias_attr):
                            expert_bias = getattr(experts, bias_attr)
                            base_layer.bias.data = expert_bias[expert_idx, :out_features].clone().to(target_dtype)
                    else:
                        weight_slice = expert_weight_slice[:, out_features:]
                        base_layer.weight.data = weight_slice.T.clone().to(target_dtype)
                        if hasattr(experts, bias_attr):
                            expert_bias = getattr(experts, bias_attr)
                            base_layer.bias.data = expert_bias[expert_idx, out_features:].clone().to(target_dtype)
                
                expert_key = f"experts.{expert_idx}.{proj_name}"
                moe_layer.experts_[expert_key] = LoraLinear(
                    base_layer,
                    config,
                    (weights[lora_a_key], weights[lora_b_key]),
                )
                experts_injected += 1
    
    print(f" ✓ Total expert LoRA layers injected: {experts_injected}")


def inject_mixlora_into_gptoss(
    model: PreTrainedModel,
    config: MixLoraConfig,
    weights: Optional[Dict[str, torch.Tensor]] = None,
) -> PreTrainedModel:
    """Inject MixLoRA adapter into GPT-OSS model."""
    
    print("\n" + "="*60)
    print("INJECTING MIXLORA INTO GPT-OSS MODEL")
    print("="*60)
    
    model_type = model.config.model_type
    config.model_type_ = model_type
    model._mixlora_config = config
    
    if weights is None:
        weights = {}
    
    print(f"Model type: {model_type}")
    print(f"Number of layers: {len(model.model.layers)}")
    print(f"Adapter name: {config.adapter_name_}")
    print(f"Number of MixLoRA experts: {config.num_experts_}")
    print(f"Top-k: {config.top_k_}")
    
    print("\nInspecting GPT-OSS architecture:")
    sample_layer = model.model.layers[0]
    sample_mlp = sample_layer.mlp
    
    print(f"  MLP type: {type(sample_mlp)}")
    print(f"  MLP attributes: {[a for a in dir(sample_mlp) if not a.startswith('_')]}")
    
    if hasattr(sample_mlp, 'experts'):
        experts = sample_mlp.experts
        print(f"  Experts type: {type(experts)}")
        print(f"  Experts attributes: {[a for a in dir(experts) if not a.startswith('_')]}")
        
        if hasattr(sample_mlp, 'num_experts'):
            print(f"  Native num_experts: {sample_mlp.num_experts}")
        if hasattr(sample_mlp, 'experts_per_token'):
            print(f"  Native experts_per_token: {sample_mlp.experts_per_token}")
    
    print()
    
    successful_attn = 0
    successful_mlp = 0
    
    for idx, layer in enumerate(model.model.layers):
        print(f"Processing Layer {idx}:")
        
        if hasattr(layer, 'attn'):
            try:
                _inject_gptoss_attn_module(idx, layer.attn, config, weights)
                successful_attn += 1
            except Exception as e:
                print(f"  ❌ Error injecting attention: {e}")
                import traceback
                traceback.print_exc()
        elif hasattr(layer, 'self_attn'):
            try:
                _inject_gptoss_attn_module(idx, layer.self_attn, config, weights)
                successful_attn += 1
            except Exception as e:
                print(f"  ❌ Error injecting attention: {e}")
                import traceback
                traceback.print_exc()
        
        if hasattr(layer, 'mlp'):
            try:
                _inject_gptoss_mlp_module(idx, layer.mlp, config, weights)
                successful_mlp += 1
            except Exception as e:
                print(f"  ❌ Error injecting MLP: {e}")
                import traceback
                traceback.print_exc()
        
        print()
    
    print("="*60)
    print(f"INJECTION COMPLETE")
    print(f"  Attention modules: {successful_attn}/{len(model.model.layers)}")
    print(f"  MLP modules: {successful_mlp}/{len(model.model.layers)}")
    print("="*60 + "\n")
    
    return model


def initialize_mixlora_weights_for_gptoss(
    model: PreTrainedModel,
    config: MixLoraConfig,
) -> Dict[str, torch.Tensor]:
    """Initialize MixLoRA weights for GPT-OSS model."""
    weights = {}
    num_layers = len(model.model.layers)
    
    print("Initializing MixLoRA weights for GPT-OSS...")
    use_aug = getattr(config, 'use_augmentation', False)
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        attn = layer.attn if hasattr(layer, 'attn') else layer.self_attn
        
        attn_projs = []
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "out"]:
            if hasattr(attn, proj_name):
                attn_projs.append(proj_name)
        
        print(f"  Layer {layer_idx}: Found attention projections: {attn_projs}")
        
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if proj_name not in config.target_modules_ or not config.target_modules_[proj_name]:
                continue
            if not hasattr(attn, proj_name):
                continue
                
            base_layer = getattr(attn, proj_name)
            
            if hasattr(base_layer, 'in_features'):
                in_features = base_layer.in_features
                out_features = base_layer.out_features
            else:
                weight_shape = base_layer.weight.shape
                in_features = weight_shape[1]
                out_features = weight_shape[0]
            
            layer_prefix = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"
            weights[f"{layer_prefix}.lora_A.weight"] = torch.randn(
                config.lora_r_, in_features
            ) * 0.01
            weights[f"{layer_prefix}.lora_B.weight"] = torch.zeros(
                out_features, config.lora_r_
            )
        
        mlp = layer.mlp
        
        hidden_size = 2880
        if hasattr(mlp, 'norm') and hasattr(mlp.norm, 'num_features'):
            hidden_size = mlp.norm.num_features
        
        gate_weight = torch.randn(config.num_experts_, hidden_size) * config.router_init_range_
        weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = gate_weight
        
        if hasattr(mlp, 'experts'):
            experts = mlp.experts
            
            if hasattr(experts, 'gate_up_proj'):
                gate_up = experts.gate_up_proj
                
                gate_up_shape = gate_up.shape
                print(f"  Layer {layer_idx}: gate_up_proj shape: {gate_up_shape}")
                
                in_features = gate_up_shape[1]
                out_features_total = gate_up_shape[2]
                intermediate_size = out_features_total // 2
                
                for proj_name in ['gate_proj', 'up_proj']:
                    if proj_name not in config.target_modules_ or not config.target_modules_[proj_name]:
                        continue
                    
                    out_features = intermediate_size
                    
                    for expert_idx in range(config.num_experts_):
                        layer_prefix = f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
                        weights[f"{layer_prefix}.lora_A.weight"] = torch.randn(
                            config.lora_r_, in_features
                        ) * 0.01
                        weights[f"{layer_prefix}.lora_B.weight"] = torch.zeros(
                            out_features, config.lora_r_
                        )
            
            if hasattr(experts, 'down_proj'):
                down = experts.down_proj
                
                down_shape = down.shape
                print(f"  Layer {layer_idx}: down_proj shape: {down_shape}")
                
                in_features = down_shape[1]
                out_features = down_shape[2]
                
                proj_name = 'down_proj'
                if proj_name in config.target_modules_ and config.target_modules_[proj_name]:
                    for expert_idx in range(config.num_experts_):
                        layer_prefix = f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
                        weights[f"{layer_prefix}.lora_A.weight"] = torch.randn(
                            config.lora_r_, in_features
                        ) * 0.01
                        weights[f"{layer_prefix}.lora_B.weight"] = torch.zeros(
                            out_features, config.lora_r_
                        )
        
        if use_aug:
            hidden_size = 2880
            if hasattr(layer.mlp, 'norm'): 
                hidden_size = layer.mlp.norm.num_features
            
            w1 = torch.empty(hidden_size // 4, hidden_size)
            b1 = torch.zeros(hidden_size // 4)
            nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
            
            weights[f"mixlora.layers.{layer_idx}.mlp.augment_gating.0.weight"] = w1
            weights[f"mixlora.layers.{layer_idx}.mlp.augment_gating.0.bias"] = b1
            
            w2 = torch.empty(1, hidden_size // 4)
            b2 = torch.zeros(1)
            nn.init.xavier_normal_(w2, gain=0.01)
            nn.init.constant_(b2, 5.0)
            
            weights[f"mixlora.layers.{layer_idx}.mlp.augment_gating.2.weight"] = w2
            weights[f"mixlora.layers.{layer_idx}.mlp.augment_gating.2.bias"] = b2
    
    print(f"Initialized {len(weights)} weight tensors")
    return weights


# ============================================================================
# WEIGHT EXTRACTION AND SAVING
# ============================================================================

def extract_mixlora_weights_from_model(
    model: PreTrainedModel,
    config: MixLoraConfig,
) -> Dict[str, torch.Tensor]:
    """Extract MixLoRA weights from the model."""
    weights = {}
    num_layers = len(model.model.layers)
    
    print(f"Extracting weights for {num_layers} layers...")
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        attn_module = getattr(layer, 'self_attn', getattr(layer, 'attn', None))
        if attn_module:
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if proj_name in config.target_modules_ and hasattr(attn_module, proj_name):
                    proj_layer = getattr(attn_module, proj_name)
                    if isinstance(proj_layer, LoraLinear):
                        layer_prefix = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"
                        weights[f"{layer_prefix}.lora_A.weight"] = proj_layer.lora_A.weight.detach().cpu().clone()
                        weights[f"{layer_prefix}.lora_B.weight"] = proj_layer.lora_B.weight.detach().cpu().clone()
        
        mlp = layer.mlp
        if hasattr(mlp, "mixlora_moes") and config.adapter_name_ in mlp.mixlora_moes:
            moe_layer = mlp.mixlora_moes[config.adapter_name_]
            
            gate_key = f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"
            weights[gate_key] = moe_layer.gate_.detach().cpu().clone()
            
            if getattr(moe_layer, 'use_augmentation_', False):
                if moe_layer.gating_network_ is not None:
                    gating_state = moe_layer.gating_network_.state_dict()
                    for k, v in gating_state.items():
                        key_name = f"mixlora.layers.{layer_idx}.mlp.augment_gating.{k}"
                        weights[key_name] = v.detach().cpu().clone()
                        
                if moe_layer.augmentator_ is not None:
                    aug_state = moe_layer.augmentator_.state_dict()
                    for k, v in aug_state.items():
                        key_name = f"mixlora.layers.{layer_idx}.mlp.augmentator.{k}"
                        weights[key_name] = v.detach().cpu().clone()
            
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if proj_name in config.target_modules_:
                    for expert_idx in range(config.num_experts_):
                        expert_key = f"experts.{expert_idx}.{proj_name}"
                        if expert_key in moe_layer.experts_:
                            lora_layer = moe_layer.experts_[expert_key]
                            layer_prefix = f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
                            weights[f"{layer_prefix}.lora_A.weight"] = lora_layer.lora_A.weight.detach().cpu().clone()
                            weights[f"{layer_prefix}.lora_B.weight"] = lora_layer.lora_B.weight.detach().cpu().clone()
    
    return weights


def save_mixlora_adapter(
    model: PreTrainedModel,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
    save_directory: str,
):
    """Save MixLoRA adapter weights and config."""
    os.makedirs(save_directory, exist_ok=True)
    
    config_dict = config.export()
    
    if getattr(config, 'use_augmentation', False):
        config_dict['use_augmentation'] = True
        config_dict['alpha_scale'] = float(getattr(config, 'alpha_scale', 0.1))
        config_dict['beta_scale'] = float(getattr(config, 'beta_scale', 0.1))
        config_dict['act_fn'] = getattr(config, 'act_fn_', 'silu')
    
    with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    torch.save(weights, os.path.join(save_directory, "adapter_model.bin"))
    print(f"✅ Adapter saved to {save_directory}")
    print(f"  - Config includes Augmentation: {config_dict.get('use_augmentation', False)}")
    print(f"  - Total tensors saved: {len(weights)}")


def verify_injection(model: PreTrainedModel, config: MixLoraConfig) -> bool:
    """Verify that MixLoRA was properly injected."""
    
    print("\n" + "="*60)
    print("VERIFYING MIXLORA INJECTION")
    print("="*60)
    
    num_layers = len(model.model.layers)
    attn_lora_count = 0
    mlp_moe_count = 0
    expert_lora_count = 0
    
    for idx, layer in enumerate(model.model.layers):
        attn = layer.attn if hasattr(layer, 'attn') else layer.self_attn
        
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, proj_name):
                proj = getattr(attn, proj_name)
                if isinstance(proj, LoraLinear):
                    attn_lora_count += 1
        
        if hasattr(layer.mlp, 'mixlora_moes'):
            if config.adapter_name_ in layer.mlp.mixlora_moes:
                mlp_moe_count += 1
                moe_layer = layer.mlp.mixlora_moes[config.adapter_name_]
                expert_lora_count += len(moe_layer.experts_)
    
    print(f"Attention LoRA layers: {attn_lora_count}")
    print(f"MLP MoE layers: {mlp_moe_count}/{num_layers}")
    print(f"Expert LoRA adapters: {expert_lora_count}")
    
    if mlp_moe_count == 0:
        print("\n⚠️  WARNING: NO MLP MoE layers were injected!")
        print("This means MixLoRA is NOT working properly.")
        return False
    elif mlp_moe_count == num_layers:
        print("\n✅ All MLP layers have MixLoRA MoE!")
    else:
        print(f"\n⚠️  WARNING: Only {mlp_moe_count}/{num_layers} MLP layers have MixLoRA")
    
    if expert_lora_count == 0:
        print("\n⚠️  WARNING: NO expert LoRA adapters were injected!")
        print("MixLoRA routing exists but no LoRA adapters on experts.")
        return False
    
    print("="*60 + "\n")
    
    return mlp_moe_count == num_layers and expert_lora_count > 0


def print_trainable_parameters_detailed(model: PreTrainedModel) -> bool:
    """Print detailed breakdown of trainable parameters."""
    
    print("\n" + "="*60)
    print("DETAILED TRAINABLE PARAMETERS")
    print("="*60)
    
    categories = {
        'attention_lora': 0,
        'mlp_expert_lora': 0,
        'moe_gates': 0,
        'aug_gating': 0,
        'other_trainable': 0,
        'frozen': 0,
    }
    
    trainable_names = []
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        name_lower = name.lower()
        
        if not param.requires_grad:
            categories['frozen'] += param_count
            continue
        
        if 'augment_gating' in name_lower:
            categories['aug_gating'] += param_count
            trainable_names.append(f"  🔹 Aug Gate:  {name} ({param_count:,})")
        elif ('attn' in name_lower or 'self_attn' in name_lower) and 'lora' in name_lower:
            categories['attention_lora'] += param_count
            trainable_names.append(f"  Attn LoRA:  {name} ({param_count:,})")
        elif 'mlp' in name_lower and 'lora' in name_lower:
            categories['mlp_expert_lora'] += param_count
            if len(trainable_names) < 5 or "layer.0." in name: 
                trainable_names.append(f"  MLP LoRA:   {name} ({param_count:,})")
            elif len(trainable_names) == 5:
                trainable_names.append(f"  ... (hiding details for other MLP LoRA layers) ...")
        elif ('gate' in name_lower or 'router' in name_lower) and ('moe' in name_lower or 'mixlora' in name_lower):
            categories['moe_gates'] += param_count
            trainable_names.append(f"  MoE Gate:   {name} ({param_count:,})")
        else:
            categories['other_trainable'] += param_count
            trainable_names.append(f"  OTHER:      {name} ({param_count:,})")
    
    print("\nSample Trainable Parameters:")
    print("-" * 60)
    for line in trainable_names[:25]:
        print(line)
    if len(trainable_names) > 25:
        print(f"  ... and {len(trainable_names) - 25} more trainable tensors")
    print("-" * 60)
    
    print("\nSummary Statistics:")
    total_params = sum(categories.values())
    trainable_params = total_params - categories['frozen']
    
    for category, count in categories.items():
        if count > 0:
            pct = 100 * count / total_params if total_params > 0 else 0
            status = "TRAINABLE" if category != 'frozen' else "FROZEN"
            print(f"  {category:20s}: {count:>12,} ({pct:>6.4f}%) [{status}]")
    
    print("-" * 60)
    print(f"  {'TOTAL PARAMS':20s}: {total_params:>12,}")
    print(f"  {'TRAINABLE':20s}: {trainable_params:>12,} ({100*trainable_params/total_params:.4f}%)")
    print("="*60 + "\n")
    
    success = True
    
    if categories['mlp_expert_lora'] == 0:
        print("❌ ERROR: No MLP expert LoRA parameters are trainable!")
        success = False
        
    if categories['moe_gates'] == 0:
        print("❌ ERROR: No MoE gate parameters are trainable!")
        success = False
        
    if categories['aug_gating'] > 0:
        print("✅ SUCCESS: Augmentation Gating Network is injected and trainable.")
    else:
        print("ℹ️ NOTE: No Augmentation Gating parameters found.")
    
    return success
