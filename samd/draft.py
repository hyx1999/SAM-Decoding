import torch
from typing import List, Tuple, Dict
from enum import Enum
from collections import namedtuple

from .samd_config import SamdConfig
from .sam import DynSAM, StaticSAM
from .tree_model import TokenRecycle
from profile_utils import profile_decorator

# from transformers import LlamaTokenizer
# tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained('/data/models/vicuna-7b-v1.3')

class CandidateType(str, Enum):
    sequence = "sequence"
    tree = "tree"

Candidates = namedtuple('Candidates', ['type', 'tokens', 'candidate_tokens'])

TOPK = 8

class DraftModel:
    
    def __init__(self,
        config: SamdConfig,
        sam_dyn: DynSAM | None = None,
        sam_static: StaticSAM | None = None,
        tree_model: TokenRecycle | None = None
    ) -> None:
        self.config = config
        self.sam_dyn = sam_dyn if sam_dyn is not None else DynSAM(config.n_predicts)
        self.sam_static = sam_static if sam_static is not None else StaticSAM(config.n_predicts)
        self.tree_model = tree_model if tree_model is not None else TokenRecycle(config.tree)
        
        self.sam_dyn.n_predicts = config.n_predicts
        self.sam_static.n_predicts = config.n_predicts
        self.len_bias = config.len_bias
        self.len_threshold = config.len_threshold
    
    def logits_to_topk(self, logits: torch.Tensor) -> List[List[int]]:
        topk_nest = logits.topk(k=TOPK).indices.cpu().tolist()
        return topk_nest
    
    def reset(self):
        self.sam_dyn.reset()
        self.sam_static.reset()
        self.tree_model.reset()

    def lookup(self, start_token: int):
        pred_dyn, match_dyn = self.sam_dyn.lookup(start_token)
        pred_static, match_static = self.sam_static.lookup(start_token)
        match_static -= self.len_bias
        if match_dyn >= match_static:
            pred, len = pred_dyn, match_dyn
        else:
            pred, len = pred_static, match_static        
        if len >= self.len_threshold:
            return CandidateType.sequence, [start_token] + pred
        else:
            return CandidateType.tree, self.tree_model.lookup(start_token)
    
    @profile_decorator("draft.update")
    def update(self, 
        tokens: List[int],
        tree_tokens: torch.Tensor,
        tree_logits: torch.Tensor,
    ):
        self.sam_dyn.add_tokens(tokens)        
        self.sam_static.transfer_tokens(tokens)
        self.tree_model.update(
            tree_tokens.cpu().tolist(),
            self.logits_to_topk(tree_logits)
        )
