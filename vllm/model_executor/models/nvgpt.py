from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import NVGPTConfig
from torch import Tensor, Size
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE, PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]

from einops import rearrange
from torch import einsum

_shape_t = Union[int, List[int], Size]
class NVGPTLayerNorm1P(torch.nn.LayerNorm):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, self.weight + 1., self.bias, self.eps) 
    
class NVGPTMLP(torch.nn.Module):
    def __init__(
        self,
        config: NVGPTConfig,
    ):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(config.hidden_size, 2 * config.ffn_hidden_size, bias=config.bias, gather_output=False, perform_initialization=False)
        self.dense_4h_to_h = RowParallelLinear(config.ffn_hidden_size, config.hidden_size, bias=config.bias, input_is_parallel=True, perform_initialization=False)        
        self.activation_func = SiluAndMul()

    def forward(self, x):
        #print(f'input: {x.shape}\t\tweight: {self.dense_h_to_4h.weight.shape}')
        intermediate_parallel, _ = self.dense_h_to_4h(x)
        #print(f'intermediate:{intermediate_parallel.shape}\t\tx: {x.shape}')
        x = self.activation_func(intermediate_parallel)
        x, _ = self.dense_4h_to_h(x)
        return x

class NVGPTAttention(torch.nn.Module):

    def __init__(
        self,
        config: NVGPTConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.scaling = self.hidden_size_per_attention_head**-0.5

        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())        
        
        assert self.num_attention_heads % tensor_model_parallel_world_size == 0

        self.num_heads = (self.num_attention_heads //
                          tensor_model_parallel_world_size)
        
        self.query_key_value = ColumnParallelLinear(
            self.hidden_size,
            3 * self.num_attention_heads * self.hidden_size_per_attention_head,
            bias=False,
            gather_output=False,
            perform_initialization=False,
        )
        self.dense = RowParallelLinear(
            self.num_attention_heads * self.hidden_size_per_attention_head,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
        )

        rotary_dim = self.hidden_size_per_attention_head
        assert 0 < config.rotary_percentage <= 1
        if config.rotary_percentage < 1:
            rotary_dim = int(rotary_dim * config.rotary_percentage)

        self.attn = PagedAttentionWithRoPE(self.num_heads,
                                           self.hidden_size_per_attention_head,
                                           self.scaling,
                                           rotary_dim=rotary_dim)
                
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)        

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        q, k, v = torch.chunk(qkv, 3, dim=-1)          

        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, q, k, v, k_cache, v_cache, input_metadata, cache_event)        
        output, _ = self.dense(attn_output)
        return output

class NVGPTDecoderLayer(torch.nn.Module):

    def __init__(self, config: NVGPTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attention = NVGPTAttention(config=config)
        self.mlp = NVGPTMLP(config=config)
        
        self.input_layernorm = NVGPTLayerNorm1P(config.hidden_size, eps=config.layernorm_eps)
        self.post_attention_layernorm = NVGPTLayerNorm1P(config.hidden_size, eps=config.layernorm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
class NVGPTModel(torch.nn.Module):

    def __init__(self, config: NVGPTConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            perform_initialization=False)
        self.layers = nn.ModuleList([
            NVGPTDecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.final_layernorm = NVGPTLayerNorm1P(config.hidden_size, eps=config.layernorm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:      
        hidden_states = self.embedding(input_ids)
        for i in range(len(self.layers)):

            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

class NVGPTForCausalLM(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = NVGPTModel(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            gather_output=False,
                                            perform_initialization=False)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = ["embedding.weight", "lm_head.weight", "query_key_value.weight","dense_h_to_4h.weight"]
    _row_parallel_weights = ["dense.weight", "dense_4h_to_h.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "rotary_pos_emb.inv_freq" in name:
                continue

            param = state_dict[name]
            if "query_key_value" in name:
                # NVGPT's fused QKV has the shape of
                # [num_heads * 3 * head_size, hidden_size], while the
                # required shape is [3 * num_heads * head_size, hidden_size].
                # Thus, we need weight conversion.

                shard_size = param.shape[0]
                start = shard_size * tensor_model_parallel_rank
                end = shard_size * (tensor_model_parallel_rank + 1)
                loaded_weight = loaded_weight[start:end]

                num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // num_heads

                loaded_weight = loaded_weight.view(-1, 3, head_size, hidden_size)
                loaded_weight = loaded_weight.transpose(0, 1)
                loaded_weight = loaded_weight.reshape(-1, hidden_size)                  

            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)
