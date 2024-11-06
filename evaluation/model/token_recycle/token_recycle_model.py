import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional, Union, List, Literal, Tuple
from types import MethodType
from transformers import LlamaForCausalLM
from .token_recycle_config import TokenRecycleConfig, ForwardState, ForwardType
from .utils import (
    CandidateType,
    TokenRecycleGenerationConfig,
    gen_buffers,
    gen_candidates,
    eval_posterior,
)
from .cache import SamdCache, SamdStaticCache
from .draft import DraftModel
from .attn_patch import attn_patch_dict
from profile_utils import profile_decorator

Outputs = namedtuple('Outputs', ['output_ids', 'decode_tokens', 'decode_steps', 'accepet_length_per_step'])

TOPK = 8

class TokenRecycleModel(nn.Module):
    
    def __init__(self,
        samd_config: TokenRecycleConfig,
        lm: LlamaForCausalLM,
        draft: DraftModel,
        eos_token_id: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        super().__init__()
        self.samd_config = samd_config
        self.gen_config: TokenRecycleGenerationConfig | None = None
        self.eos_token = eos_token_id

        self.lm = lm
        self.draft = draft
        self.dtype = dtype
        self.device = device
        
        # buffers
        self.seq_position_ids: torch.Tensor | None = None
        self.tree_attn_mask: torch.Tensor | None = None
        self.tree_position_ids: torch.Tensor | None = None
        self.tree_retrieve_indices: torch.Tensor | None = None
        self.tree_size: int = len(samd_config.tree)
        
        # buffers
        self.cache: Optional[SamdCache | SamdStaticCache] = None
        self.forward_state = ForwardState(None)
        
        self.init_buffers()
        self.register_forward_patch()

    def register_forward_patch(self):
        for module_name, module in self.lm.named_modules():
            if type(module) in attn_patch_dict:
                setattr(module, "tree_attn_mask", self.tree_attn_mask)
                setattr(module, "forward_state", self.forward_state)
                for fn_name, fn in attn_patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
                    print("setattr {}.{}".format(module_name, fn_name))
    
    def init_buffers(self):
        buffers = gen_buffers(self.samd_config, self.device)
        self.seq_position_ids = buffers["seq_position_ids"]
        self.tree_attn_mask = buffers["tree_attn_mask"]
        self.tree_position_ids = buffers["tree_position_ids"]
        self.tree_retrieve_indices = buffers["tree_retrieve_indices"]
    
    def reset_cache(
        self,
    ):
        max_cache_len = self.gen_config.max_cache_len + self.tree_size
        if self.cache is not None and self.cache.max_cache_len == max_cache_len:
            self.cache.reset()
        else:
            self.cache = SamdStaticCache(
                self.lm.config.num_hidden_layers, 
                self.lm.config.num_attention_heads,
                self.lm.config.num_key_value_heads,
                self.lm.config.hidden_size,
                max_batch_size=1,
                max_cache_len=max_cache_len,
                device=self.device,
                dtype=self.dtype,
            )
    
    def check_cache(self):
        if isinstance(self.cache, SamdStaticCache):
            return self.cache.get_seq_length() + self.tree_size <= self.cache.max_cache_len
        else:
            return True
    
    def logits_to_topk(self, logits: torch.Tensor) -> List[List[int]]:
        topk_nest = logits.topk(k=TOPK).indices.cpu().tolist()
        return topk_nest
    
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
        self.draft.update(input_ids_list, self.logits_to_topk(logits.squeeze(0)))
        self.cache.set_length()
        return logits[:, -1:]  # [1, 1, D]
    
    @profile_decorator("decode")
    def decode(self, logits: torch.Tensor, length: int):
        candidates = gen_candidates(
            logits,
            self.tree_retrieve_indices,
            self.draft,
            self.samd_config, 
            self.gen_config, 
            self.device
        )
        # print("candidates:", candidates)
        if candidates.type == CandidateType.sequence:
            self.forward_state.forward_type = ForwardType.seq_decode
            position_ids = self.seq_position_ids + length
        else:
            self.forward_state.forward_type = ForwardType.tree_decode
            position_ids = self.tree_position_ids + length
        input_ids = candidates.tokens
        tree_logits = self.lm(
            input_ids=input_ids, 
            position_ids=position_ids,
            past_key_values=self.cache,
        ).logits
        if candidates.type == CandidateType.sequence:
            candidate_logits = tree_logits
            candidate_indices = None
        else:
            candidate_logits = tree_logits.squeeze(0)[self.tree_retrieve_indices]
            candidate_indices = self.tree_retrieve_indices

        best_candidate, accept_length = eval_posterior(candidate_logits, candidates.candidate_tokens, self.gen_config)
        new_logits, new_tokens = self.update_state(
            input_ids,
            tree_logits,
            best_candidate, 
            accept_length,
            candidate_logits,
            candidates.candidate_tokens,
            candidate_indices
        )
        return new_logits, new_tokens

    @profile_decorator("update_state")
    def update_state(self,
        tree_tokens: torch.Tensor,
        tree_logits: torch.Tensor,
        best_candidate: torch.Tensor, 
        accept_length: torch.Tensor,
        candidate_logits: torch.Tensor,
        candiate_tokens: torch.Tensor,
        candidate_indices: torch.Tensor | None = None,
    ):
        logits = candidate_logits[best_candidate]
        tokens = candiate_tokens[best_candidate]
        logits = logits[:accept_length]
        tokens = tokens[:accept_length].cpu().tolist()
                
        if candidate_indices is not None:
            indices = candidate_indices[best_candidate][:accept_length]
        else:
            indices = None

        # print("indices:", indices)
        # print("accepet_length:", accept_length)
        
        self.draft.update(
            tree_tokens.squeeze(0).cpu().tolist(), 
            self.logits_to_topk(tree_logits.squeeze(0))
        )
        self.cache.select_indices(indices, accept_length.item())
        
        logits = logits[-1].view(1, 1, -1)

        return logits, tokens
    
    @torch.inference_mode()
    def generate(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: TokenRecycleGenerationConfig | None = None, 
    ) -> Outputs:
        if generation_config is None:
            generation_config = TokenRecycleGenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"  # [1, N]
        
        self.cache = SamdCache(self.lm.config.num_hidden_layers)
        # self.reset_cache()
        self.draft.reset()
        
        input_ids_list = input_ids.squeeze(0).cpu().tolist()
        logits = self.prefill(input_ids, attention_mask, input_ids_list)
        
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
            # if self.check_cache() == False:
            #     break
        input_ids_list = [input_ids_list[:input_length + generation_config.max_new_tokens]]
        return Outputs(input_ids_list, decode_tokens, decode_steps, accepet_length_per_step)
