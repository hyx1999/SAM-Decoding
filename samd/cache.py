import numpy as np
import torch
from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig
from typing import List

class SamdStaticCache(StaticCache):
    
    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__(config, max_batch_size, max_cache_len, device, dtype)
        self.cache_positions: np.ndarray = np.array([0] * config.num_hidden_layers)
        self.devcie = device
    
    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.cache_positions[layer_idx]
    
    def reset(self):
        super().reset()
        self.cache_positions.fill(0)
    
    @torch.no_grad()
    def select_path(self, path: List[int]):
        accept_length = len(path)
        path = torch.tensor(path, dtype=torch.long, device=self.devcie).add_(self.cache_positions[0])
        for layer_idx in range(len(self.key_cache)):
            cache_position = self.cache_positions[layer_idx]
            self.key_cache[layer_idx][:, :, cache_position:cache_position + accept_length] \
                = self.key_cache[layer_idx][:, :, path]
            self.value_cache[layer_idx][:, :, cache_position:cache_position + accept_length] \
                = self.key_cache[layer_idx][:, :, path]
            self.cache_positions[layer_idx] += accept_length
