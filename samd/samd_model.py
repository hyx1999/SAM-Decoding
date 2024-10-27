import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import namedtuple
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
from .attn_patch import attn_patch_dict

Outputs = namedtuple('Outputs', ['output_ids', 'decode_tokens', 'decode_steps', 'accepet_length'])

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
    
    def init_sam_hot_state(self, output_ids):
        self.sam.init_hot_state(output_ids)
    
    def prefill(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None):
        logits = self.lm(input_ids=input_ids, position_ids=position_ids).logits
        return logits[:, -1:]  # [1, 1, D]
    
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

        topk = tree_logits[path].topk(k=self.samd_config.k)
        path_topk = list(zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()))

        self.sam.update_samping_state(state_indices, path_topk)
        self.sam.update_hot_state(path, path_topk)
        self.cache.select_path(path)
        
        new_logits = tree_logits[0, path[-1]].view(1, 1, -1)
        new_tokens = new_tokens.tolist()

        return new_logits, new_tokens
    
    @torch.inference_mode()
    def generate(self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None, 
    ) -> Outputs:
        if generation_config is None:
            generation_config = GenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"  # [1, N]
        
        self.sam.reset_state()
        
        output_ids = input_ids.view(-1).cpu().tolist()        

        self.init_cache(
            input_ids.shape[-1],
            generation_config,
        )
        
        self.init_sam_hot_state(output_ids)
        
        logits = self.prefill(input_ids, position_ids)
        
        input_length = input_ids.shape[-1]
        decode_tokens = 0
        decode_steps = 0
        accepet_length = []
        for step in range(generation_config.max_new_tokens):
            logits, new_ids = self.decode(logits)
            eos_index = None
            if self.eos_token in new_ids:
                eos_index = new_ids.index(self.eos_token)
                new_ids = new_ids[:eos_index]
            output_ids.extend(new_ids)
            decode_steps += 1
            decode_tokens += len(new_ids)
            accepet_length.append(len(new_ids))
            if eos_index is not None:
                break
            if decode_tokens >= generation_config.max_new_tokens:
                break
        output_ids = output_ids[:input_length + generation_config.max_new_tokens]
        return Outputs(output_ids, decode_tokens, decode_steps, accepet_length)
