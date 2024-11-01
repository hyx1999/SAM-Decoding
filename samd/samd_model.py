import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional, Union, List, Literal, Tuple
from types import MethodType
from transformers import LlamaForCausalLM
from .samd_config import SamdConfig, ForwardState, ForwardType
from .utils import (
    SamdGenerationConfig, 
    gen_sd_buffers,
    gen_candidates,
    eval_posterior,
)
from .cache import SamdCache, SamdStaticCache
from .sam import SAM
from .attn_patch import attn_patch_dict

Outputs = namedtuple('Outputs', ['output_ids', 'decode_tokens', 'decode_steps', 'accepet_length_per_step'])

class SamdModel(nn.Module):
    
    def __init__(self,
        samd_config: SamdConfig,
        lm: LlamaForCausalLM,
        sam: SAM,
        eos_token_id: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.samd_config = samd_config
        self.gen_config: Optional[SamdGenerationConfig] = None
        self.eos_token = eos_token_id

        self.lm = lm
        self.sam = sam
        self.device = device
        self.dtype = dtype
        
        # sd buffers
        self.samd_attn_mask: Optional[torch.Tensor] = None
        self.samd_position_ids: Optional[torch.Tensor] = None
        self.tree_size: int = len(samd_config.tree)
        
        # buffers
        self.cache: Optional[SamdCache] = None
        self.forward_state = ForwardState(None)
        
        self.init_sd_buffers()
        self.register_forward_patch()
    
    def logits_to_topk(self, logits: torch.Tensor) -> List[List[Tuple[int, float]]]:
        topk = torch.softmax(logits, dim=-1).topk(k=self.samd_config.k)
        topk = [list(zip(x, y)) for x, y in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist())]
        return topk
    
    def register_forward_patch(self):
        for module_name, module in self.lm.named_modules():
            if type(module) in attn_patch_dict:
                setattr(module, "samd_attn_mask", self.samd_attn_mask)
                setattr(module, "forward_state", self.forward_state)
                for fn_name, fn in attn_patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
                    print("setattr {}.{}".format(module_name, fn_name))
    
    def init_sd_buffers(self):
        sd_buffers = gen_sd_buffers(self.samd_config.tree, self.device)
        self.samd_attn_mask = sd_buffers["samd_attn_mask"]
        self.samd_position_ids = sd_buffers["samd_position_ids"]

    def reset_cache(
        self,
        gen_config: SamdGenerationConfig, 
    ):
        max_cache_len = gen_config.max_cache_len + self.tree_size
        if self.cache is not None and self.cache.max_cache_len == max_cache_len:
            self.cache.reset()
        else:
            self.cache = SamdCache(
                self.lm.config, 
                max_batch_size=1,
                max_cache_len=max_cache_len,
                device=self.device,
                dtype=self.dtype,
            )
    
    def check_cache(self):
        return self.cache.get_seq_length() + self.tree_size <= self.cache.max_cache_len
    
    def init_sam_hot_state(self, input_ids_list):
        self.sam.init_hot_state(input_ids_list)
    
    def prefill(self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        input_ids_list: List[int],
    ):
        self.forward_state.forward_type = ForwardType.prefill
        logits = self.lm(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            past_key_values=self.cache,
        ).logits
        self.cache.set_cache_positions(input_ids.shape[-1])
        self.sam.transfer_state(input_ids_list)
        self.sam.update_sam_online(input_ids_list)
        return logits[:, -1:]  # [1, 1, D]
    
    def decode(self, logits: torch.Tensor, length: int):
        candidates = gen_candidates(logits, self.sam, self.samd_config, self.gen_config, self.eos_token, self.device)
        tree_input_ids = candidates.tree_tokens
        tree_position_ids = self.samd_position_ids + length
        tree_logits = self.lm(
            input_ids=tree_input_ids, 
            position_ids=tree_position_ids,
            past_key_values=self.cache,
        ).logits
        candidate_logits = tree_logits.squeeze(0)[candidates.candidate_indices]

        best_candidate, accept_length = eval_posterior(candidate_logits, candidates.candidate_tokens, self.gen_config)
        new_logits, new_tokens = self.update_state(
            best_candidate.item(), 
            accept_length.item(),
            candidate_logits,
            candidates.candidate_tokens,
            candidates.candidate_indices, 
            candidates.candidate_sam_indices,
        )
        return new_logits, new_tokens
    
    def update_state(self,
        best_candidate: int, 
        accept_length: int,
        candidate_logits: torch.Tensor,
        candiate_tokens: torch.Tensor, 
        candiate_indices: torch.Tensor, 
        candiate_sam_indices: List[List[int]],
    ):
        logits = candidate_logits[best_candidate]
        tokens = candiate_tokens[best_candidate]
        indices = candiate_indices[best_candidate]
        state_indices = candiate_sam_indices[best_candidate]

        assert indices[accept_length - 1].item() != -1

        logits = logits[:accept_length]
        tokens = tokens[:accept_length].cpu().tolist()
        indices = indices[:accept_length]
        state_indices = state_indices[:accept_length]
        
        self.sam.set_state_index(state_indices[-1])
        self.sam.update_sam_online(tokens)
        self.cache.post_update(indices)
        
        logits = logits[-1].view(1, 1, -1)

        return logits, tokens
    
    @torch.inference_mode()
    def generate(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: SamdGenerationConfig | None = None, 
    ) -> Outputs:
        if generation_config is None:
            generation_config = SamdGenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"  # [1, N]
        
        self.sam.reset_state()
        self.reset_cache(generation_config)
        
        input_ids_list = input_ids.squeeze(0).cpu().tolist()
        logits = self.prefill(input_ids, attention_mask, input_ids_list)
                
        self.forward_state.forward_type = ForwardType.decode
        input_length = input_ids.shape[-1]
        decode_tokens = 0
        decode_steps = 0
        accepet_length_per_step = []
        for step in range(generation_config.max_new_tokens):
            logits, new_ids = self.decode(logits, input_length + decode_tokens)
            eos_index = None
            if self.eos_token in new_ids:
                eos_index = new_ids.index(self.eos_token)
                new_ids = new_ids[:eos_index + 1]
            input_ids_list.extend(new_ids)
            decode_steps += 1
            decode_tokens += len(new_ids)
            accepet_length_per_step.append(len(new_ids))
            if eos_index is not None:
                break
            if decode_tokens >= generation_config.max_new_tokens:
                break
            if not self.check_cache():
                break
        input_ids_list = [input_ids_list[:input_length + generation_config.max_new_tokens]]
        return Outputs(input_ids_list, decode_tokens, decode_steps, accepet_length_per_step)
