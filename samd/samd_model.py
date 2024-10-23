import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal
from types import MethodType

from .samd_config import SamdConfig
from .utils import (
    GenerationConfig, 
    gen_sd_buffers,
    gen_candidates,
    eval_posterior,
)
from .cache import SamdStaticCache
from .sam import SAM
from attn_patch import attn_patch_dict

class SamdModel:
    
    def __init__(self,
        samd_config: SamdConfig,
        lm: nn.Module,
        sam: SAM,
        eos_token_id: int,
    ) -> None:
        self.samd_config = samd_config
        self.gen_config: Optional[GenerationConfig] = None
        self.eos_token = eos_token_id

        self.lm = lm
        self.sam = sam
        self.device = next(self.lm.parameters()).device
        
        # sd buffers
        self.samd_attn_mask: Optional[torch.Tensor] = None
        self.samd_position_ids: Optional[torch.Tensor] = None
        
        # buffers
        self.cache: Optional[SamdStaticCache] = None
        
        self.init_sd_buffers()
        self.register_forward_patch()
    
    def register_forward_patch(self):
        for name, module in self.lm.named_modules():
            if type(module) in attn_patch_dict:
                setattr(module, "samd_attn_mask", self.samd_attn_mask)
                for fn_name, fn in attn_patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
    
    def init_sd_buffers(self):
        sd_buffers = gen_sd_buffers(self.samd_config.tree)
        self.samd_attn_mask = sd_buffers["samd_attn_mask"]
        self.samd_position_ids = sd_buffers["samd_position_ids"]

    @torch.no_grad()
    def init_cache(
        self,
        input_length: int,
        generation_config: GenerationConfig, 
    ):
        max_len = input_length + generation_config.max_new_tokens + self.samd_position_ids.shape[1]
        self.cache = SamdStaticCache(
            self.lm.config, 
            max_batch_size=1,
            max_cache_len=max_len,
            device=self.device
        )
    
    def prefill(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None):
        logits = self.lm(input_ids=input_ids, position_ids=position_ids).logits
        return logits
    
    def decode(self, logits: torch.Tensor):
        candidates = gen_candidates(logits, self.sam, self.samd_config, self.gen_config, self.eos_token, self.device)
        tree_input_ids = candidates.tree_tokens
        tree_position_ids = self.samd_position_ids + self.length
        tree_logits = self.lm(input_ids=tree_input_ids, position_ids=tree_position_ids)
        best_candidate, accept_length = eval_posterior(tree_logits, candidates.candidate_tokens, self.gen_config)
        new_logits, new_tokens = self.update_state(
            tree_logits, 
            best_candidate.item(), 
            accept_length.item(),
            candidates.candidate_tokens,
            candidates.candiate_paths, 
            candidates.candiate_indices,
        )
        return new_logits, new_tokens
    
    def update_state(self,
        tree_logits: torch.Tensor,
        best_candidate: int, 
        accept_length: int,
        candiate_tokens: torch.Tensor, 
        candiate_paths: List[List[int]], 
        candiate_indices: List[List[int]],
    ):
        new_tokens = candiate_tokens[best_candidate]
        path = candiate_paths[best_candidate]
        state_indices = candiate_indices[best_candidate]
        while path[accept_length - 1] == -1:
            accept_length -= 1
        new_tokens = new_tokens[:accept_length]
        path = path[:accept_length]
        state_indices = state_indices[:accept_length]
        path_topk = tree_logits[path].topk(k=self.samd_config.k).indices.cpu().tolist()

        self.sam.update_samping_state(state_indices, path_topk)      
        self.cache.select_path(path)
        
        new_logits = tree_logits[path[-1]]
        new_tokens = new_tokens.tolist()

        return new_logits, new_tokens
    
    def generate(self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        generation_config: Optional[GenerationConfig] = None, 
    ):
        if generation_config is None:
            generation_config = GenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"
        
        self.init_cache(
            input_ids.shape[-1],
            generation_config,
        )
        
        logits = self.prefill(input_ids, position_ids)
        
        gen_tokens = []
        for step in range(generation_config.max_new_tokens):
            logits, new_tokens = self.decode(logits)
            eos_index = None
            if self.eos_token in new_tokens:
                eos_index = new_tokens.index(self.eos_token)
                new_tokens = new_tokens[:eos_index]
            gen_tokens.extend(new_tokens)
            if eos_index is not None:
                break
        return gen_tokens
