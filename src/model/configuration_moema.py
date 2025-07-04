# coding=utf-8
# Copyright 2023 Mixtral AI and the HuggingFace Inc. team. All rights reserved.
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

"""
Moema model configuration

This file is based on MixtralConfig (by Mixtral AI & HuggingFace) and LlamaConfig (by Meta AI).
It combines the structure and key hyperparameters of both Mixtral's Mixture-of-Experts (MoE) and Llama-style transformer models.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.modeling_rope_utils import rope_config_validation

logger = logging.get_logger(__name__)

class MoemaConfig(PretrainedConfig):
    """
    Configuration class for the Moema model.

    This class is inspired by MixtralConfig and LlamaConfig.
    It provides configuration for MoE (Mixture of Experts) and Llama-style transformer models.

    Args:
        vocab_size (int): Size of the model vocabulary.
        hidden_size (int): Dimension of the hidden representations.
        intermediate_size (int): Dimension of the MLP (feedforward) layer.
        num_hidden_layers (int): Number of transformer layers.
        num_attention_heads (int): Number of attention heads.
        num_key_value_heads (int): Number of key/value heads for multi-query/grouped attention.
        head_dim (int): Dimension of each attention head.
        hidden_act (str): Activation function for the MLP.
        max_position_embeddings (int): Maximum sequence length.
        initializer_range (float): Standard deviation for weight initialization.
        rms_norm_eps (float): Epsilon for RMSNorm.
        use_cache (bool): Whether to use caching for generation.
        pad_token_id (int): Padding token ID.
        bos_token_id (int): Beginning-of-sequence token ID.
        eos_token_id (int): End-of-sequence token ID.
        tie_word_embeddings (bool): Whether to tie input and output embeddings.
        rope_theta (float): RoPE base theta value.
        attention_bias (bool): Whether to use bias in attention layers.
        mlp_bias (bool): Whether to use bias in MLP layers.
        sliding_window (int or None): Sliding window size for attention.
        attention_dropout (float): Dropout rate for attention.
        num_experts_per_tok (int): Number of experts per token (top-k routing).
        num_local_experts (int): Total number of experts (MoE).
        output_router_logits (bool): Whether to output router logits.
        router_aux_loss_coef (float): Coefficient for the router auxiliary loss.
        router_jitter_noise (float): Jitter noise for router during training.
        rope_scaling (dict or None): RoPE scaling configuration.
        **kwargs: Additional keyword arguments.
    """
    model_type = "moema"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        attention_bias=False,
        mlp_bias=False,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # For backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    
    @classmethod
    def from_llama_config(
        cls,
        llama_config,
        num_experts: int = 6,
        num_experts_per_tok: int = 2,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.01,
        router_jitter_noise: float = 0.1,
        sliding_window=None,
    ):
        """
        Create a MoemaConfig from a LlamaConfig.

        Args:
            llama_config (PretrainedConfig): Llama model config instance.
            num_experts (int): Number of experts for MoE.
            num_experts_per_tok (int): Top-k experts per token.
            output_router_logits (bool): Whether to output router logits.
            router_aux_loss_coef (float): Coefficient for router auxiliary loss.
            router_jitter_noise (float): Jitter noise for the router.
            sliding_window (int or None): Sliding window size for attention.

        Returns:
            MoemaConfig: The converted MoemaConfig instance.
        """
        return cls(
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            intermediate_size=llama_config.intermediate_size,
            num_hidden_layers=llama_config.num_hidden_layers,
            num_attention_heads=llama_config.num_attention_heads,
            num_key_value_heads=llama_config.num_key_value_heads,
            head_dim=llama_config.head_dim,
            hidden_act=llama_config.hidden_act,
            max_position_embeddings=llama_config.max_position_embeddings,
            initializer_range=llama_config.initializer_range,
            rms_norm_eps=llama_config.rms_norm_eps,
            use_cache=llama_config.use_cache,
            pad_token_id=llama_config.pad_token_id,
            bos_token_id=llama_config.bos_token_id,
            eos_token_id=llama_config.eos_token_id,
            tie_word_embeddings=llama_config.tie_word_embeddings,
            rope_theta=llama_config.rope_theta,
            attention_bias=llama_config.attention_bias,
            mlp_bias=llama_config.mlp_bias,
            sliding_window=sliding_window,
            attention_dropout=llama_config.attention_dropout,
            num_experts_per_tok=num_experts_per_tok,
            num_local_experts=num_experts,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            router_jitter_noise=router_jitter_noise,
            rope_scaling=llama_config.rope_scaling,
        )

__all__ = ["MoemaConfig"]