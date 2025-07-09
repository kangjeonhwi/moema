import torch
import random
import numpy as np
import os
import argparse 

from src.model.modeling_llama import LlamixDecoderLayer
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
import torch.nn as nn

def set_seed(seed: int):
    """재현성을 위해 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initialize_weights(size, std=0.02, mean=0.0, device="cpu", dtype=torch.float32):
    """정규분포를 따르는 가중치를 초기화하는 함수"""
    return torch.normal(mean=mean, std=std, size=size, device=device, dtype=dtype)

def shuffle_and_process_ffn(
    tensor,
    perm,
    target_size,
    is_down_proj,
    ffn_init_ratio,
    method,
):
    """FFN 가중치를 섞고 'drop' 또는 'noise' 방식에 따라 처리하는 함수"""
    if is_down_proj:
        shuffled = tensor.index_select(1, perm[:target_size]).clone()
    else:
        shuffled = tensor.index_select(0, perm[:target_size]).clone()

    init_size = int(target_size * ffn_init_ratio)
    if init_size == 0:
        return shuffled

    init_indices = torch.randperm(target_size)[:init_size]

    if method == "drop":
        if is_down_proj:
            init_part = shuffled[:, init_indices]
            init_mean = init_part.mean().item()
            init_std = init_part.std().item()
            init_tensor = initialize_weights(
                (tensor.size(0), init_size), std=init_std, mean=init_mean, device=tensor.device, dtype=tensor.dtype
            )
            shuffled[:, init_indices] = init_tensor
        else:
            init_part = shuffled[init_indices, :]
            init_mean = init_part.mean().item()
            init_std = init_part.std().item()
            init_tensor = initialize_weights(
                (init_size, tensor.size(1)), std=init_std, mean=init_mean, device=tensor.device, dtype=tensor.dtype
            )
            shuffled[init_indices, :] = init_tensor
    elif method == "noise":
        if is_down_proj:
            init_tensor = initialize_weights(
                (tensor.size(0), init_size), std=0.02, mean=0, device=tensor.device, dtype=tensor.dtype
            )
            shuffled[:, init_indices] += init_tensor
        else:
            init_tensor = initialize_weights(
                (init_size, tensor.size(1)), std=0.02, mean=0, device=tensor.device, dtype=tensor.dtype
            )
            shuffled[init_indices, :] += init_tensor
    else:
        raise ValueError(f"Unknown method: {method}")

    return shuffled

def convert_llama_to_llamix(
    model: "LlamaForCausalLM",
    method: str = "naive",
    ffn_init_ratio: float = 0.5,
    seed: int = 42,
) -> "LlamaForCausalLM":
    set_seed(seed)
    config = model.config
    new_layers = []

    for i, old_layer in enumerate(model.model.layers):
        new_layer = LlamixDecoderLayer(config=config, layer_idx=i)

        new_layer.self_attn.load_state_dict(old_layer.self_attn.state_dict())
        new_layer.input_layernorm.load_state_dict(old_layer.input_layernorm.state_dict())
        new_layer.post_attention_layernorm.load_state_dict(old_layer.post_attention_layernorm.state_dict())
        new_layer.mlp1.load_state_dict(old_layer.mlp.state_dict())

        if method == "naive":
            new_layer.mlp2.load_state_dict(old_layer.mlp.state_dict())
        elif method in ["noise", "drop"]:
            old_mlp_state = old_layer.mlp.state_dict()
            new_mlp2_state = {}
            intermediate_size = old_layer.mlp.gate_proj.out_features
            perm = torch.randperm(intermediate_size, device=model.device) 
      
            for key, tensor in old_mlp_state.items():
                if "weight" in key and tensor.dim() == 2:
                    is_down_proj = "down_proj" in key 
                    shuffled_tensor = shuffle_and_process_ffn(
                        tensor=tensor,
                        perm=perm,
                        target_size=intermediate_size,
                        is_down_proj=is_down_proj,
                        ffn_init_ratio=ffn_init_ratio,
                        method=method,
                    )
                    new_mlp2_state[key] = shuffled_tensor
                else:
                    new_mlp2_state[key] = tensor.clone()
            
            new_layer.mlp2.load_state_dict(new_mlp2_state)
        else:
            raise ValueError(f"Unknown method: {method}")

        new_layers.append(new_layer)
    model.model.layers = torch.nn.ModuleList(new_layers)
    return model

def save_model(model, output_dir, model_name):
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving '{model_name}' model to '{save_path}' ...")
    model.save_pretrained(save_path)
    print(f"'{model_name}' model saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Llama 모델을 Llamix 모델로 변환하고 저장합니다.")
    parser.add_argument(
        "--llama_path",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="변환할 기본 Llama 모델의 경로 또는 Hugging Face 모델 ID"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['naive', 'drop', 'noise'],
        default='drop',
        help="mlp2 어댑터 초기화 방법"
    )
    parser.add_argument(
        "--ffn_init_ratio",
        type=float,
        default=0.5,
        help="'drop' 또는 'noise' 메소드에서 사용할 FFN 초기화 비율"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/MoE/moema2/moema/output/twined_models",
        help="변환된 모델을 저장할 기본 디렉토리"
    )
    args = parser.parse_args()

    DTYPE = torch.bfloat16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 기본 Llama 모델 로드
    print(f"Loading base model from: {args.llama_path}")
    model = LlamaForCausalLM.from_pretrained(args.llama_path).to(DEVICE, dtype=DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(args.llama_path)
    if tokenizer.pad_token is None:
        print("Pad token not set. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 모델 변환 실행
    print(f"Converting model using method='{args.method}' with ffn_init_ratio={args.ffn_init_ratio}...")
    converted_model = convert_llama_to_llamix(
        model=model,
        method=args.method,
        ffn_init_ratio=args.ffn_init_ratio,
        seed=42  # 시드 고정
    )

    base_model_name = args.llama_path.split("/")[-1]
    save_model_name = f"{base_model_name}_{args.method}_{args.ffn_init_ratio}"
    save_model(converted_model, args.output_path, save_model_name)
    
    tokenizer_save_path = os.path.join(args.output_path, save_model_name)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Tokenizer saved to '{tokenizer_save_path}'")


if __name__ == "__main__":
    main()