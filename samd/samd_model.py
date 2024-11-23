import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional, Union, List, Literal, Tuple, Dict
from types import MethodType
from transformers import LlamaForCausalLM
from .samd_config import SamdConfig, ForwardState, ForwardType, MaskState
from .utils import (
    OptionalTensor,
    CandidateType,
    SamdGenerationConfig,
    gen_candidates,
    eval_posterior,
)
from .cache import SamdCache, SamdStaticCache
from .draft import DraftModel
from .model_patch import patch_dict, attn_patch_dict
from profile_utils import profile_decorator, profile_accept_length

Outputs = namedtuple('Outputs', ['output_ids', 'decode_tokens', 'decode_steps', 'accepet_length_per_step'])

class SamdModel(nn.Module):
    
    def __init__(self,
        samd_config: SamdConfig,
        lm: LlamaForCausalLM,
        draft: DraftModel,
        eos_token_id: int,
        dtype: torch.dtype,
        device: str,
        stop_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.samd_config = samd_config
        self.gen_config: SamdGenerationConfig = None
        self.eos_token = eos_token_id
        self.stop_token = stop_token_id

        self.lm = lm
        self.draft = draft
        self.dtype = dtype
        self.device = device
        
        # buffers
        self.seq_position_ids: torch.Tensor = None
        self.tree_attn_mask: torch.Tensor = None
        self.tree_position_ids: torch.Tensor = None
        self.tree_retrieve_indices: torch.Tensor = None
        
        # buffers
        self.cache: SamdCache = None
        self.forward_state = ForwardState(None)
        self.mask_state = MaskState(None)
        
        self.init_buffers()
        self.register_forward_patch()

    def register_forward_patch(self):
        for module_name, module in self.lm.named_modules():
            module_name = "root" if module_name == "" else "root.{}".format(module_name)
            if type(module) in patch_dict:
                for fn_name, fn in patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
                    print("setattr {} -> {}".format(module_name, fn_name))
            if type(module) in attn_patch_dict:
                for fn_name, fn in attn_patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
                    setattr(module, "mask_state", self.mask_state)
                    setattr(module, "forward_state", self.forward_state)
                    print("attn setattr {} -> {}".format(module_name, fn_name))

    
    def init_seq_position_ids(self):
        return torch.tensor(
            range(0, self.samd_config.n_predicts + 1), 
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
    
    def init_buffers(self):
        self.seq_position_ids = self.init_seq_position_ids()
        buffers = self.draft.tree_model.gen_buffers()
        self.tree_attn_mask = buffers["tree_attn_mask"]
        self.tree_position_ids = buffers["tree_position_ids"]
        self.tree_retrieve_indices = buffers["tree_retrieve_indices"]
        self.mask_state.set_state(self.tree_attn_mask)
    
    def update_buffers(self, buffers_kwargs: Dict[str, Optional[torch.Tensor]]):
        self.tree_attn_mask = buffers_kwargs.get("tree_attn_mask", self.tree_attn_mask)
        self.tree_position_ids = buffers_kwargs.get("tree_position_ids", self.tree_position_ids)
        self.tree_retrieve_indices = buffers_kwargs.get("tree_retrieve_indices", self.tree_retrieve_indices)
        self.mask_state.set_state(self.tree_attn_mask)
    
    @profile_decorator("SamdModel.prefill")
    def prefill(self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        self.forward_state.forward_type = ForwardType.prefill
        outputs = self.lm(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            past_key_values=self.cache,
        )
        logits = outputs.logits
        last_hidden_states = outputs.last_hidden_states \
            if self.samd_config.use_last_hidden_states else None
        last_hidden_states = OptionalTensor(last_hidden_states).apply(
            lambda x: x.squeeze(0)
        ).data
        self.draft.update(
            tokens=input_ids.squeeze(0),
            last_hidden_states=last_hidden_states,
            tree_tokens=input_ids.squeeze(0),
            tree_logits=logits.squeeze(0)
        )
        self.cache.set_length()
        return logits[:, -1:]  # [1, 1, D]
    
    @profile_decorator("SamdModel.decode")
    def decode(self, logits: torch.Tensor, length: int):
        candidates = gen_candidates(
            logits,
            self.tree_retrieve_indices,
            self.draft,
            self.samd_config, 
            self.gen_config, 
            self.device
        )
        self.update_buffers(candidates.buffers_kwargs)
        if candidates.type == CandidateType.sequence_dyn \
            or candidates.type == CandidateType.sequence_static:
            self.forward_state.forward_type = ForwardType.seq_decode
            position_ids = self.seq_position_ids + length
        else:
            self.forward_state.forward_type = ForwardType.tree_decode
            position_ids = self.tree_position_ids + length
        input_ids = candidates.tokens
        outputs = self.lm(
            input_ids=input_ids, 
            position_ids=position_ids,
            past_key_values=self.cache,
        )
        tree_logits = outputs.logits
        # print("tree_logits.shape:", tree_logits.shape)
        if self.samd_config.use_last_hidden_states:
            tree_last_hidden_states = OptionalTensor(outputs.last_hidden_states)
        else:
            tree_last_hidden_states = OptionalTensor(None)
        if candidates.type == CandidateType.sequence_dyn \
            or candidates.type == CandidateType.sequence_static:
            candidate_logits = tree_logits
            candidate_last_hidden_states = tree_last_hidden_states
            candidate_indices = OptionalTensor(None)
        else:
            candidate_logits = tree_logits.squeeze(0)[self.tree_retrieve_indices]
            candidate_last_hidden_states = tree_last_hidden_states.apply(
                lambda x: x.squeeze(0)[self.tree_retrieve_indices]
            )
            candidate_indices = OptionalTensor(self.tree_retrieve_indices)

        best_candidate, accept_length = eval_posterior(candidate_logits, candidates.candidate_tokens, self.gen_config)
        new_logits, new_tokens = self.update_state(
            input_ids.squeeze(0),
            tree_logits.squeeze(0),
            best_candidate, 
            accept_length,
            candidate_logits,
            candidates.candidate_tokens,
            candidate_indices,
            candidate_last_hidden_states,
        )
        # print("new_tokens:\n{}".format(new_tokens))
        return new_logits, new_tokens

    @profile_decorator("SamdModel.update_state")
    def update_state(self,
        tree_tokens: torch.Tensor,
        tree_logits: torch.Tensor,
        best_candidate: torch.Tensor, 
        accept_length: torch.Tensor,
        candidate_logits: torch.Tensor,
        candiate_tokens: torch.Tensor,
        candidate_indices: OptionalTensor,
        candidate_last_hidden_states: OptionalTensor,
    ):
        logits = candidate_logits[best_candidate][:accept_length]
        tokens = candiate_tokens[best_candidate][:accept_length]
        
        indices: Optional[torch.Tensor] = candidate_indices.apply(
            lambda x: x[best_candidate][:accept_length]
        ).data
        last_hidden_states: Optional[torch.Tensor] = candidate_last_hidden_states.apply(
            lambda x: x[best_candidate][:accept_length]
        ).data
        
        self.draft.update(
            tokens=tokens, 
            last_hidden_states=last_hidden_states,
            tree_tokens=tree_tokens,
            tree_logits=tree_logits,
        )
        self.cache.select_indices(indices, accept_length.item())
        
        logits = logits[-1].view(1, 1, -1)

        return logits, tokens.tolist()
    
    @torch.inference_mode()
    def generate(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        generation_config: SamdGenerationConfig = None, 
    ) -> Outputs:
        if generation_config is None:
            generation_config = SamdGenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"  # [1, N]

        # self.reset_static_cache()        
        self.cache = SamdCache(self.lm.config.num_hidden_layers)  # use dynamic cache

        self.draft.reset()
        
        input_ids_list = input_ids.squeeze(0).tolist()
        logits = self.prefill(input_ids, attention_mask)
        
        input_length = input_ids.shape[-1]
        decode_tokens = 0
        decode_steps = 0
        accepet_length_per_step = []
        for step in range(generation_config.max_new_tokens):
            if input_length + decode_tokens + self.samd_config.n_predicts >= generation_config.max_cache_len:
                break
            logits, new_ids = self.decode(logits, input_length + decode_tokens)
            eos_index = None
            if self.eos_token in new_ids:
                eos_index = new_ids.index(self.eos_token)
                new_ids = new_ids[:eos_index + 1]
            elif self.stop_token is not None and self.stop_token in new_ids:
                eos_index = new_ids.index(self.stop_token)
                new_ids = new_ids[:eos_index + 1]
            input_ids_list.extend(new_ids)
            decode_steps += 1
            decode_tokens += len(new_ids)
            accepet_length_per_step.append(len(new_ids))
            profile_accept_length("lookup", len(new_ids))
            if eos_index is not None:
                break
            if decode_tokens >= generation_config.max_new_tokens:
                break
        input_ids_list = [input_ids_list[:input_length + generation_config.max_new_tokens]]
        return Outputs(input_ids_list, decode_tokens, decode_steps, accepet_length_per_step)


"""
    def reset_static_cache(self):
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
"""