# Copyright [2025] 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is developed to upcycle a dense LLaMA-style model into a
# Mixture-of-Experts (MoE) model, named Moema.
#
# The core upcycling logic, particularly the 'drop' and 'noise' methods,
# is heavily inspired by and based on the implementation found in the
# Drop-Upcycling repository. We extend our sincere gratitude to the original
# authors for their valuable work.
#
# For more details on the original methodology and for license information
# pertaining to the referenced code, please visit:
# https://github.com/Taishi-N324/Drop-Upcycling/tree/main

import argparse
import logging
import random
import re
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from ..model.configuration_moema import MoemaConfig
from ..model.modeling_moema import MoemaForCausalLM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_gate_weights(size, method):
    if method == "torch_rand":
        return torch.rand(size)
    elif method == "torch_rand_mean0":
        weights = torch.rand(size)
        weights_mean = weights.mean()
        return weights - weights_mean
    elif method == "torch_normal_002":
        return torch.normal(mean=0, std=0.02, size=size)
    elif method == "torch_normal_028":
        return torch.normal(mean=0, std=0.2886751345948129, size=size)
    elif method == "torch_rand_002":
        weights = torch.rand(size)
        weights_mean = weights.mean()
        return (weights - weights_mean) * 0.02 * (12**0.5)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def initialize_weights(size, method, std=0.02, mean=0):
    logger.info(f"Initializing weights: method={method}, std={std}, mean={mean}")
    if method == "torch_normal":
        return torch.normal(mean=mean, std=std, size=size)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def shuffle_and_process_ffn(
    tensor,
    perm,
    target_size,
    is_down_proj,
    layer_idx,
    expert_idx,
    ffn_init_ratio,
    upcycling_method,
):
    if is_down_proj:
        original_size = tensor.size(1)
        shuffled = tensor.index_select(1, perm[:target_size])
    else:
        original_size = tensor.size(0)
        shuffled = tensor.index_select(0, perm[:target_size])

    init_size = int(target_size * ffn_init_ratio)
    if init_size == 0:
        logger.warning(
            f"Layer {layer_idx}, Expert {expert_idx}: ffn_init_ratio ({ffn_init_ratio}) is too low, "
            f"resulting in 0 neurons to initialize. Skipping modification."
        )
        return shuffled

    init_indices = torch.randperm(target_size)[:init_size]

    if upcycling_method == "drop":
        if is_down_proj:
            init_part = shuffled[:, init_indices]
            init_mean = init_part.mean().item()
            init_std = init_part.std().item()
            init_tensor = initialize_weights(
                (tensor.size(0), init_size), "torch_normal", std=init_std, mean=init_mean
            ).to(dtype=torch.bfloat16)
            shuffled[:, init_indices] = init_tensor
        else:
            init_part = shuffled[init_indices, :]
            init_mean = init_part.mean().item()
            init_std = init_part.std().item()
            init_tensor = initialize_weights(
                (init_size, tensor.size(1)), "torch_normal", std=init_std, mean=init_mean
            ).to(dtype=torch.bfloat16)
            shuffled[init_indices, :] = init_tensor

    elif upcycling_method == "noise":
        if is_down_proj:
            init_tensor = initialize_weights(
                (tensor.size(0), init_size), "torch_normal", std=0.02, mean=0
            ).to(dtype=torch.bfloat16)
            shuffled[:, init_indices] += init_tensor
        else:
            init_tensor = initialize_weights(
                (init_size, tensor.size(1)), "torch_normal", std=0.02, mean=0
            ).to(dtype=torch.bfloat16)
            shuffled[init_indices, :] += init_tensor

    logger.info(
        f"Layer {layer_idx}, Expert {expert_idx}, Method: {upcycling_method}, "
        f"{'Down_proj' if is_down_proj else 'Gate_proj/Up_proj'}: "
        f"Original size: {original_size}, Target size: {target_size}, Modified size: {init_size}"
    )
    return shuffled


def upcycle_model(
    source_model_path: str,
    output_path: str,
    upcycling_method: str,
    num_experts: int,
    num_experts_per_tok: int,
    gate_init_method: str,
    ffn_init_ratio: float,
    output_router_logits: bool,
    router_aux_loss_coef: float,
    router_jitter_noise: float,
    seed: int,
    sliding_window=None,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    set_seed(seed)

    logger.info("Loading the source LLaMA model...")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_path, torch_dtype=torch_dtype
    ).eval()
    logger.info("Source model loaded successfully.")

    logger.info("Initializing the target MoE model structure...")
    target_config = MoemaConfig.from_llama_config(
        source_model.config,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        output_router_logits=output_router_logits,
        router_aux_loss_coef=router_aux_loss_coef,
        router_jitter_noise=router_jitter_noise,
        sliding_window=sliding_window,
    )
    target_model = MoemaForCausalLM(target_config).to(dtype=torch_dtype).eval()
    logger.info("Target MoE model initialized successfully.")
    logger.info("---------------------------\nTarget model configuration:\n%s\n---------------------------", target_config)

    source_params = dict(source_model.named_parameters())
    
    with torch.no_grad():
        logger.info("Copying shared parameters (non-FFN/MoE)...")
        for name, param in tqdm(target_model.named_parameters(), desc="Copying shared layers"):
            if "block_sparse_moe" not in name:
                if name in source_params:
                    param.data.copy_(source_params[name].data)
                else:
                    logger.warning(f"Parameter '{name}' not found in the source model. Skipping.")
        
        logger.info(f"Initializing MoE layers using '{upcycling_method}' method...")
        for layer_idx, layer in enumerate(tqdm(target_model.model.layers, desc=f"Processing MoE layers ({upcycling_method})")):

            gate_weight = layer.block_sparse_moe.gate.weight
            initialized_gate = initialize_gate_weights(gate_weight.shape, gate_init_method)
            gate_weight.data.copy_(initialized_gate)
            
            source_gate_proj = source_params[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
            source_down_proj = source_params[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
            source_up_proj = source_params[f"model.layers.{layer_idx}.mlp.up_proj.weight"]

            if upcycling_method == "naive":
                for expert_idx, expert in enumerate(layer.block_sparse_moe.experts):
                    expert.w1.weight.data.copy_(source_gate_proj.data)
                    expert.w3.weight.data.copy_(source_up_proj.data)
                    expert.w2.weight.data.copy_(source_down_proj.data)
                logger.info(f"Layer {layer_idx}: Naively copied FFN weights to all {num_experts} experts.")
            
            elif upcycling_method in ["noise", "drop"]:
                target_intermediate_size = target_config.intermediate_size
                for expert_idx, expert in enumerate(layer.block_sparse_moe.experts):
                    perm = torch.randperm(target_intermediate_size)
                    
                    for source_tensor, expert_w in [(source_gate_proj, expert.w1), (source_up_proj, expert.w3)]:
                        shuffled_tensor = shuffle_and_process_ffn(
                            tensor=source_tensor, perm=perm, target_size=target_intermediate_size,
                            is_down_proj=False, layer_idx=layer_idx, expert_idx=expert_idx,
                            ffn_init_ratio=ffn_init_ratio, upcycling_method=upcycling_method
                        )
                        expert_w.weight.data.copy_(shuffled_tensor)
   
                    shuffled_w2 = shuffle_and_process_ffn(
                        tensor=source_down_proj, perm=perm, target_size=target_intermediate_size,
                        is_down_proj=True, layer_idx=layer_idx, expert_idx=expert_idx,
                        ffn_init_ratio=ffn_init_ratio, upcycling_method=upcycling_method
                    )
                    expert.w2.weight.data.copy_(shuffled_w2)
            else:
                raise ValueError(f"Unknown upcycling method: {upcycling_method}")

    logger.info("Model upcycling complete.")
    logger.info(f"Saving the upcycled model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    target_model.save_pretrained(output_path)
    logger.info(f"Model successfully saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Upcycle a dense model to a Mixture-of-Experts (MoE) model.")
    
    parser.add_argument("--source_model_path", type=str, required=True, help="Path to the source dense model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the upcycled MoE model.")
    parser.add_argument(
        "--upcycling_method", type=str, required=True, choices=["naive", "noise", "drop"],
        help="The method for upcycling FFN layers: 'naive', 'noise', or 'drop'."
    )
    
    parser.add_argument("--num_experts", type=int, required=True, help="Number of experts in the target MoE model.")
    parser.add_argument("--num_experts_per_tok", type=int, default=2, help="Number of experts to route to for each token.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--gate_init_method", type=str, default="torch_rand",
        choices=["torch_rand", "torch_rand_mean0", "torch_normal_002", "torch_normal_028", "torch_rand_002"],
        help="Method for initializing gate weights in the MoE layers."
    )
    parser.add_argument(
        "--ffn_init_ratio", type=float, default=0.5,
        help="For 'noise' and 'drop' methods, the ratio of FFN neurons to be modified (0.0 to 1.0)."
    )
    
    parser.add_argument("--output_router_logits", action='store_true', help="Whether to output router logits.")
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01, help="Router auxiliary loss coefficient.")
    parser.add_argument("--router_jitter_noise", type=float, default=0.1, help="Router jitter noise.")
    
    args = parser.parse_args()

    if args.upcycling_method == "naive" and args.ffn_init_ratio != 0.5:
        logger.warning(
            f"--ffn_init_ratio ({args.ffn_init_ratio}) is provided but will be ignored "
            f"because --upcycling_method is 'naive'."
        )

    upcycle_model(
        source_model_path=args.source_model_path,
        output_path=args.output_path,
        upcycling_method=args.upcycling_method,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        gate_init_method=args.gate_init_method,
        ffn_init_ratio=args.ffn_init_ratio,
        output_router_logits=args.output_router_logits,
        router_aux_loss_coef=args.router_aux_loss_coef,
        router_jitter_noise=args.router_jitter_noise,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()