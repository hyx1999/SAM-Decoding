import numpy as np
import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache, Cache
from typing import Optional, Dict, Any, Tuple
from profile_utils import profile_decorator

class SamdCache(DynamicCache):
    
    def __init__(self, num_hidden_layers: int | None = None) -> None:
        super().__init__(num_hidden_layers)
        self.cache_length = 0
    
    def set_length(self):
        self.cache_length = self.get_seq_length()
    
    @profile_decorator("SamdCache.select_indices")
    def select_indices(self,
        indices: torch.Tensor | None = None,
        accept_length: int = 1,
    ):
        start = self.cache_length
        if indices is not None:
            select_indices = start + indices
        else:
            select_indices = None
        for data in self.key_cache + self.value_cache:
            if select_indices is not None:
                select_indices = select_indices.to(data.device)
                tgt = data.index_select(-2, select_indices)
                dst = data.narrow(-2, start, accept_length)
                dst.copy_(tgt)                
        self.cache_length += accept_length
        self.crop(self.cache_length)


class SamdStaticCache(Cache):
    
    def __init__(self, 
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        max_batch_size: int,
        max_cache_len: int,
        device=None, 
        dtype=None
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.cur_length = 0
        self.new_length = 0
        self.kv_data = torch.zeros(
            num_hidden_layers * 2,
            max_batch_size,
            num_key_value_heads,
            max_cache_len,
            hidden_size // num_attention_heads,
            device=device,
            dtype=dtype
        )
        self.devcie = device
        self.dtype = dtype
    
    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.cur_length
    
    def get_max_cache_shape(self) -> int:
        return self.max_cache_len
    
    def reorder_cache(self, beam_idx):
        raise NotImplementedError
    
    def reset(self):
        self.cur_length = 0
        self.new_length = 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.cur_length
        length = key_states.shape[-2]
        self.new_length = start + length
        self.kv_data[2 * layer_idx + 0]\
            .narrow(-2, start, length)\
            .copy_(key_states)
        self.kv_data[2 * layer_idx + 1]\
            .narrow(-2, start, length)\
            .copy_(value_states)

        k_out = self.kv_data[2 * layer_idx + 0].narrow(-2, 0, start + length)
        v_out = self.kv_data[2 * layer_idx + 1].narrow(-2, 0, start + length)
        return k_out, v_out
    
    def select_indices(self,
        indices: torch.Tensor | None = None,
        accept_length: int = 1,
    ):
        start = self.cur_length
        if indices is not None:
            select_indices = start + indices
        else:
            select_indices = None
        if select_indices is not None:
            tgt = self.kv_data.index_select(-2, select_indices)
            dst = self.kv_data.narrow(-2, start, accept_length)
            dst.copy_(tgt)
        self.cur_length += accept_length

    def set_length(self):
        self.cur_length = self.new_length
