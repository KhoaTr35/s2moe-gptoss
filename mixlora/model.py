import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN

from .config import MixLoraConfig
from .lora_linear import LoraLinear
from .utils import infer_device


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
        
        # 1. Trainable Instance Norm (affine=True -> Learnable Gamma/Beta)
        self.instance_norm = nn.InstanceNorm1d(feature_size, eps=eps, affine=True)
        
        # 2. Trainable Noise Scaling
        self.alpha_scale = nn.Parameter(torch.tensor(float(alpha_scale)))
        self.beta_scale = nn.Parameter(torch.tensor(float(beta_scale)))

    def forward(self, features, *args, **kwargs):
        """
        Args:
            features: Tensor of shape [Batch, Seq, Hidden] or [Batch*Seq, Hidden]
        """
        # 1. Auto-Device Placement (Fix lỗi device mismatch)
        target_device = features.device
        if self.alpha_scale.device != target_device:
            self.to(target_device)

        input_dtype = features.dtype
        
        # 2. Shape Handling
        is_flat = features.dim() == 2
        if is_flat:
            total_tokens, hidden_dim = features.shape
            # Treat batch as 1, Channels=Hidden, Seq=TotalTokens
            x = features.view(1, total_tokens, hidden_dim)
        else:
            x = features

        x_permuted = x.permute(0, 2, 1) # [Batch, Hidden, Seq]

        # 3. SAFETY CHECK: Skip if Sequence Length is <= 1 
        # (InstanceNorm sẽ crash nếu chỉ có 1 token để tính std)
        if x_permuted.size(2) <= 1:
            return features

        # 4. Apply Norm & Noise
        # Cast sang FP32 để tính toán ổn định
        x_permuted_f32 = x_permuted.to(torch.float32)
        x_norm = self.instance_norm(x_permuted_f32)
        
        alpha_noise = torch.randn_like(x_norm) * self.alpha_scale
        beta_noise = torch.randn_like(x_norm) * self.beta_scale
        
        augmented_norm = (1 + alpha_noise) * x_norm + beta_noise

        # 5. Restore
        augmented = augmented_norm.to(input_dtype)
        augmented = augmented.permute(0, 2, 1)
        
        if is_flat:
            augmented = augmented.view(-1, hidden_dim)
            
        return augmented

    @classmethod
    def code(cls) -> str:
        return 'normal_gaussian_trainable'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _slice_tensor(
    data: torch.Tensor,
    slice: torch.Tensor,
    dtype: torch.dtype,
    last_value: Optional[torch.Tensor] = None,
):
    if last_value is None:
        # for macOS debugging, please uncomment this line
        # assert data.dtype in (torch.float, torch.int, torch.bool)
        return data[None, slice].reshape(-1, data.shape[-1]).to(dtype)
    else:
        return last_value


_compatible_model_types = {
    "llama": "_llama_forward",
    "gemma": "_llama_forward",
    "gemma2": "_llama_forward",
    "qwen2": "_llama_forward",
    "mistral": "_llama_forward",
    "phi": "_phi_forward",
    "phi3": "_phi3_forward",
    "gpt_oss": "_gptoss_forward",
    "gpt-oss": "_gptoss_forward",
    "gpt_oss_20b": "_gptoss_forward",
}


class MixLoraSparseMoe(torch.nn.Module):
    """MixLoRA Sparse Mixture of Experts with Dual-Path Augmentation."""

    def __init__(
        self,
        base_layer: torch.nn.Module,
        config: MixLoraConfig,
    ) -> None:
        super().__init__()
        self.config = config

        # Determine compute dtype
        self.dtype_: torch.dtype = getattr(config, 'dtype_', getattr(config, 'dtype', torch.float32))

        # Main router parameter: [NumExperts, HiddenDim]
        self.gate_: torch.nn.Parameter = None

        # Setup base layer and experts dictionary
        object.__setattr__(self, 'base_layer_', base_layer)
        self.experts_: Dict[str, LoraLinear] = {}

        # Configuration parameters (experts, top-k, regularization)
        self.num_experts_: int = getattr(config, 'num_experts_', getattr(config, 'num_experts', 4))
        default_topk = getattr(config, 'top_k_', getattr(config, 'top_k', 2))
        self.moe_topk_: int = getattr(config, 'moe_top_k_', getattr(config, 'moe_top_k', default_topk))
        self.jitter_noise_: float = getattr(config, 'jitter_noise_', getattr(config, 'jitter_noise', 0.0))

        # Augmentation path components (used during training)
        self.use_augmentation_: bool = getattr(config, 'use_augmentation', False)
        self.augmentator_: Optional[NormalAugmenter] = None
        self.gating_network_: Optional[nn.Module] = None

        # Resolve activation function
        act_fn_name = getattr(config, 'act_fn_', getattr(config, 'act_fn', 'silu'))
        if isinstance(act_fn_name, str):
            act_obj = ACT2FN.get(act_fn_name, torch.nn.SiLU)
            self.act_fn_ = act_obj() if isinstance(act_obj, type) else act_obj
        else:
            self.act_fn_ = act_fn_name

    def _ensure_device_and_dtype(self, target_tensor: torch.Tensor):
        """Aligns internal state (gate, aux modules) with input tensor for device/dtype consistency."""
        target_device = target_tensor.device
        target_dtype = target_tensor.dtype

        # 1. Align Main Router Gate (nn.Parameter)
        if self.gate_ is not None:
            if self.gate_.device != target_device or self.gate_.dtype != target_dtype:
                self.gate_.data = self.gate_.data.to(device=target_device, dtype=target_dtype)

        # 2. Align Augmentation Components
        if self.use_augmentation_:
            # Align Gating Network
            if self.gating_network_ is not None:
                first_param = next(self.gating_network_.parameters(), None)
                if first_param is not None and (first_param.device != target_device or first_param.dtype != target_dtype):
                    self.gating_network_ = self.gating_network_.to(device=target_device, dtype=target_dtype)
            
            # Align Augmentator device
            if self.augmentator_ is not None and hasattr(self.augmentator_, 'alpha_scale'):
                if self.augmentator_.alpha_scale.device != target_device:
                    self.augmentator_.to(target_device) 

    def _compute_router_logits(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates logits, applies jitter, selects Top-K experts, and re-normalizes weights."""
        
        # Ensure gate weight matches input
        gate_w = self.gate_
        if gate_w.device != hidden_states.device or gate_w.dtype != hidden_states.dtype:
            gate_w = gate_w.to(device=hidden_states.device, dtype=hidden_states.dtype)
            
        router_logits = torch.matmul(hidden_states, gate_w.T)
        
        # Apply jitter noise during training
        if self.training and self.jitter_noise_ > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise_
            
        # Softmax and Top-K selection
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.moe_topk_, dim=-1)
        
        # Re-normalize Top-K weights
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        return router_logits, routing_weights, selected_experts

    def _process_moe(self, hidden_states: torch.Tensor):
        """Standard MoE forward: Routing, Expert execution, and Weighted combination."""
        input_dtype = hidden_states.dtype
        
        # 1. Routing
        router_logits, routing_weights, selected_experts = self._compute_router_logits(hidden_states)
        
        # 2. Expert Mask (for token dispatching)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts_).permute(2, 1, 0)
        
        # 3. Execute Experts
        expert_outputs = self._gptoss_forward(expert_mask, hidden_states, input_dtype)
        
        # 4. Combine
        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts_):
            # Find tokens routed to this expert
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0: continue
                
            current_output = expert_outputs[expert_idx]
            
            # Aggregate Top-K weights for weighted sum
            expert_map = (selected_experts[top_x] == expert_idx)
            token_weights = (routing_weights[top_x] * expert_map.float()).sum(dim=-1, keepdim=True)
            
            final_hidden_states[top_x] += current_output * token_weights

        return final_hidden_states, router_logits

    def _gptoss_forward(
        self,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        input_dtype: torch.dtype,
    ):
        """
        GPT-OSS MLP forward with MixLoRA experts.
        
        Architecture specifics:
        - Weights shape: [num_experts, in_features, out_features] (No transpose needed for matmul)
        - Activation: SiLU (implied by SwiGLU structure)
        - Empty Experts: Returns (0, hidden) tensor to prevent propagation errors.
        """
        # 1. Validation & Setup
        base_experts = self.base_layer_.experts
        
        # Validation: Ensure activation function is present
        # Ưu tiên dùng act_fn_ đã inject, nếu không có thì fallback về SiLU (Safety net)
        act_fn = self.act_fn_ if hasattr(self, 'act_fn_') and self.act_fn_ is not None else torch.nn.functional.silu

        # 2. Get Projections
        if hasattr(base_experts, 'gate_up_proj'):
            gate_up_proj = base_experts.gate_up_proj
            gate_up_bias = getattr(base_experts, 'gate_up_proj_bias', None)
        else:
            # Trường hợp model không đúng chuẩn GPT-OSS
            raise ValueError("Critical: 'gate_up_proj' not found in GPT-OSS experts.")
            
        down_proj = base_experts.down_proj
        down_bias = getattr(base_experts, 'down_proj_bias', None)
        
        final_expert_states = []
        
        # 3. Expert Loop
        for expert_idx in range(self.num_experts_):
            # Get indices of tokens routed to this expert
            _, top_x = torch.where(expert_mask[expert_idx])
            
            # --- ROBUST FIX: EMPTY EXPERT HANDLING ---
            if top_x.numel() == 0:
                # Trả về tensor rỗng đúng dtype và device để tránh lỗi broadcasting phía sau
                empty_tensor = torch.empty(
                    (0, hidden_states.shape[-1]), 
                    dtype=input_dtype, 
                    device=hidden_states.device
                )
                final_expert_states.append(empty_tensor)
                continue
            # -----------------------------------------
            
            # Slice input for specific tokens
            expert_hidden = _slice_tensor(hidden_states, top_x, input_dtype)
            
            # --- A. Gate-Up Projection ---
            if isinstance(gate_up_proj, torch.nn.Linear):
                gate_up_output = gate_up_proj(expert_hidden)
            else:
                # GPT-OSS Param: [num_experts, in, out]
                # Matmul: [tokens, in] @ [in, out] -> [tokens, out]
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
            
            # --- B. Activation (SwiGLU core) ---
            # act_fn(gate) * up
            act_result = act_fn(gate_output) * up_output
            
            # --- C. Down Projection ---
            if isinstance(down_proj, torch.nn.Linear):
                down_output = down_proj(act_result)
            else:
                # GPT-OSS Param: [num_experts, intermediate, hidden]
                # Matmul: [tokens, intermediate] @ [intermediate, hidden]
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

    def _llama_forward(
        self,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        input_dtype: torch.dtype,
    ):
        common_gate = self.base_layer_.gate_proj(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        common_up = self.base_layer_.up_proj(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            _, top_x = torch.where(expert_mask[expert_idx])
            lora_gate: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.gate_proj", None
            )
            lora_down: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.down_proj", None
            )
            lora_up: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.up_proj", None
            )
            if lora_gate is not None:
                lora_data = _slice_tensor(hidden_states, top_x, input_dtype)
                gate_states = lora_gate.lora_forward(
                    _slice_tensor(common_gate, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                gate_states = _slice_tensor(common_gate, top_x, input_dtype)

            if lora_up is not None:
                lora_data = _slice_tensor(hidden_states, top_x, input_dtype)
                up_states = lora_up.lora_forward(
                    _slice_tensor(common_up, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                up_states = _slice_tensor(common_up, top_x, input_dtype)

            act_result = self.act_fn_(gate_states) * up_states

            if lora_down is not None:
                final_expert_states.append(
                    lora_down.lora_forward(
                        self.base_layer_.down_proj(act_result), act_result
                    )
                )
            else:
                final_expert_states.append(self.base_layer_.down_proj(act_result))

        return final_expert_states

    def _phi_forward(
        self,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        input_dtype: torch.dtype,
    ):
        common_fc1 = self.base_layer_.fc1(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            _, top_x = torch.where(expert_mask[expert_idx])
            lora_fc1: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.fc1", None
            )
            lora_fc2: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.fc2", None
            )
            if lora_fc1 is not None:
                lora_data = _slice_tensor(hidden_states, top_x, input_dtype)
                act_result = self.act_fn_(
                    lora_fc1.lora_forward(
                        _slice_tensor(common_fc1, top_x, input_dtype), lora_data
                    )
                )
            else:
                act_result = self.act_fn_(_slice_tensor(common_fc1, top_x, input_dtype))

            if lora_fc2 is not None:
                final_expert_states.append(
                    lora_fc2.lora_forward(self.base_layer_.fc2(act_result), act_result)
                )
            else:
                final_expert_states.append(self.base_layer_.fc2(act_result))

        return final_expert_states

    def _phi3_forward(
        self,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        input_dtype: torch.dtype,
    ):
        common_gate_up = self.base_layer_.gate_up_proj(
            hidden_states.to(input_dtype)
        ).to(hidden_states.dtype)
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            _, top_x = torch.where(expert_mask[expert_idx])
            lora_gate_up: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.gate_up_proj", None
            )
            lora_down: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.down_proj", None
            )
            if lora_gate_up is not None:
                gate_up_states = lora_gate_up.lora_forward(
                    _slice_tensor(common_gate_up, top_x, input_dtype),
                    _slice_tensor(hidden_states, top_x, input_dtype),
                )
            else:
                gate_up_states = _slice_tensor(common_gate_up, top_x, input_dtype)

            gate_states, up_states = gate_up_states.chunk(2, dim=-1)
            act_result = up_states * self.act_fn_(gate_states)

            if lora_down is not None:
                final_expert_states.append(
                    lora_down.lora_forward(
                        self.base_layer_.down_proj(act_result), act_result
                    )
                )
            else:
                final_expert_states.append(self.base_layer_.down_proj(act_result))

        return final_expert_states

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        original_shape = hidden_states.shape
        # Flatten input [B, S, H] -> [B*S, H]
        hidden_states_flat = hidden_states.view(-1, original_shape[-1])
        
        # Runtime device/dtype check
        self._ensure_device_and_dtype(hidden_states_flat)
        
        # Dual Path Augmentation (Training only)
        if self.use_augmentation_ and self.training and self.augmentator_ is not None:
            # 1. Gating Score (Mix ratio)
            gate_logit = self.gating_network_(hidden_states_flat)  
            gate_score = torch.sigmoid(gate_logit)
            
            # 2. Original Path output
            output_original, logits_original = self._process_moe(hidden_states_flat)
            
            # 3. Augmented Path output
            augmented_input = self.augmentator_(hidden_states_flat)
            output_augmented, _ = self._process_moe(augmented_input)
            
            # 4. Mix: (Score * Original) + ((1 - Score) * Augmented)
            final_output = gate_score * output_original + (1 - gate_score) * output_augmented
            
            # Restore shape and return original logits
            return final_output.view(original_shape), logits_original
            
        else:
            # Standard MoE pass (Inference or Augmentation disabled)
            final_output, logits = self._process_moe(hidden_states_flat)
            return final_output.view(original_shape), logits


def _inject_attn_module(
    layer_idx: int,
    self_attn: torch.nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    for proj_name, inject in config.target_modules_.items():
        if not inject or not hasattr(self_attn, proj_name):
            continue
        base_layer = getattr(self_attn, proj_name)
        layer_prefix_name = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"
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


def _inject_mlp_module(
    layer_idx: int,
    mlp: torch.nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    moe_layer = MixLoraSparseMoe(mlp, config)
    moe_layer.gate_ = weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"].to(
        config.dtype_
    )

    if not hasattr(mlp, "mixlora_moes"):
        mlp.mixlora_moes = {}

    mlp.mixlora_moes[config.adapter_name_] = moe_layer
    mlp.forward = moe_layer.forward

    for proj_name, inject in config.target_modules_.items():
        if not inject or not hasattr(mlp, proj_name):
            continue
        base_layer = getattr(mlp, proj_name)
        for expert_idx in range(config.num_experts_):
            layer_prefix_name = (
                f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
            )
            moe_layer.experts_[f"experts.{expert_idx}.{proj_name}"] = LoraLinear(
                base_layer,
                config,
                (
                    weights[f"{layer_prefix_name}.lora_A.weight"],
                    weights[f"{layer_prefix_name}.lora_B.weight"],
                ),
            )


def inject_adapter_in_model(
    model: PreTrainedModel,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    config.model_type_ = model.config.model_type
    model._mixlora_config = config
    for idx, layer in enumerate(model.model.layers):
        _inject_attn_module(idx, layer.self_attn, config, weights)
        _inject_mlp_module(idx, layer.mlp, config, weights)


def load_adapter_weights(
    name_or_path: str,
    adapter_name: str = "default",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
):
    if not os.path.exists(name_or_path):
        name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")

    if device is None:
        device = infer_device()

    with open(
        name_or_path + os.sep + "adapter_config.json", "r", encoding="utf8"
    ) as fp:
        config = MixLoraConfig.from_config(json.load(fp))
        config.adapter_name_ = adapter_name
        config.dtype_ = dtype

    config.check()

    weights: Dict[str, torch.Tensor] = torch.load(
        name_or_path + os.sep + "adapter_model.bin",
        map_location=device,
        weights_only=True,
    )

    return config, weights


_compatible_task_types = ["CAUSAL_LM", "QUESTION_ANS"]


@dataclass
class MixLoraModelForCausalLM:
    @staticmethod
    def from_pretrained(
        name_or_path: str,
        *model_args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, MixLoraConfig]:
        config, weights = load_adapter_weights(
            name_or_path,
            adapter_name=kwargs.pop("adapter_name", "default"),
            dtype=kwargs.get("torch_dtype", torch.float32),
        )

        assert config.task_type_ in _compatible_task_types

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_, *model_args, **kwargs
        )

        inject_adapter_in_model(model, config, weights)

        return model, config
