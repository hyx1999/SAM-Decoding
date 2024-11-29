import torch
from typing import List, Tuple, Dict, Optional
from enum import Enum
from collections import namedtuple

from .samd_config import SamdConfig
from .sam import DynSAM, StaticSAM
from profile_utils import profile_decorator
from transformers import LlamaConfig, LlamaForCausalLM

# from transformers import LlamaTokenizer
# tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained('/data/models/vicuna-7b-v1.3')

class CandidateType(str, Enum):
    sequence = "sequence"
    tree = "tree"

Candidates = namedtuple('Candidates', ['type', 'tokens', 'candidate_tokens', 'buffers_kwargs'])

TOPK = 8

class DraftModel(torch.nn.Module):
    
    def __init__(self,
        config: SamdConfig,
        sam_dyn: DynSAM = None,
        sam_static: StaticSAM = None,
        lm: LlamaForCausalLM = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.sam_dyn = sam_dyn if sam_dyn is not None else DynSAM(config.n_predicts)
        self.sam_static = sam_static if sam_static is not None else StaticSAM(config.n_predicts)        
        self.sam_dyn.max_predicts = config.n_predicts
        self.sam_static.max_predicts = config.n_predicts
        self.len_bias = config.len_bias

    @profile_decorator("DraftModel.reset")        
    def reset(self):
        self.sam_dyn.reset()
        self.sam_static.reset()

    @profile_decorator("DraftModel.lookup")
    def lookup(self, start_token: int):
        index_dyn, match_dyn = self.sam_dyn.lookup(start_token)
        index_static, match_static = self.sam_static.lookup(start_token)
        match_static -= self.len_bias
        if match_dyn >= match_static:
            tree, buffers_kwargs = self.sam_dyn.gen_draft(index_dyn, match_dyn, start_token, self.device)
        else:
            tree, buffers_kwargs = self.sam_static.gen_draft(index_static, match_static, start_token, self.device)
        return (CandidateType.tree, tree, buffers_kwargs)
    
    @profile_decorator("DraftModel.update")
    def update(self, 
        tokens: Optional[torch.Tensor] = None,
    ):
        tokens_list = tokens.tolist()
        self.sam_dyn.add_tokens(tokens_list)
        self.sam_static.transfer_tokens(tokens_list)

    @profile_decorator("DraftModel.prefill_update")
    def prefill_update(self, 
        tokens: Optional[torch.Tensor] = None,
    ):
        self.update(tokens)
