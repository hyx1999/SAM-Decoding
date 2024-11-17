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
        self.sam_dyn = sam_dyn if sam_dyn is not None else DynSAM(config.n_predicts)
        self.sam_static = sam_static
        
        self.sam_dyn.n_predicts = config.n_predicts
        if self.sam_static is not None:
            self.sam_static.n_predicts = config.n_predicts
        self.len_bias = config.len_bias

    @profile_decorator("DraftModel.reset")        
    def reset(self):
        self.sam_dyn.reset()
        if self.sam_static is not None:
            self.sam_static.reset()

    @profile_decorator("DraftModel.lookup")
    def lookup(self, start_token: int):
        pred_dyn, match_dyn = self.sam_dyn.lookup(start_token)
        if self.sam_static is not None:
            pred_static, match_static = self.sam_static.lookup(start_token)
        else:
            pred_static = []
            match_static = -1
        match_static -= self.len_bias
        if match_dyn >= match_static:
            pred, len = pred_dyn, match_dyn
        else:
            pred, len = pred_static, match_static        
        return (CandidateType.sequence, [start_token] + pred, {})
    
    @profile_decorator("DraftModel.update")
    def update(self, 
        tokens: Optional[torch.Tensor] = None,
    ):
        tokens_list = tokens.tolist()
        self.sam_dyn.add_tokens(tokens_list)
        if self.sam_static is not None:
            self.sam_static.transfer_tokens(tokens_list)

    @profile_decorator("DraftModel.prefill_update")
    def prefill_update(self, 
        tokens: Optional[torch.Tensor] = None,
    ):
        self.update(tokens)
