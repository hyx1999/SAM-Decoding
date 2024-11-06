import torch
from typing import List, Tuple, Dict
from enum import Enum
from collections import namedtuple

from .token_recycle_config import TokenRecycleConfig
from .token_recycle import TokenRecycle
from profile_utils import profile_decorator

# from transformers import LlamaTokenizer
# tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained('/data/models/vicuna-7b-v1.3')

class CandidateType(str, Enum):
    sequence = "sequence"
    tree = "tree"

Candidates = namedtuple('Candidates', ['type', 'tokens', 'candidate_tokens'])

class DraftModel:
    
    def __init__(self,
        config: TokenRecycleConfig,
        tree_model: TokenRecycle | None = None
    ) -> None:
        self.config = config
        self.tree_model = tree_model if tree_model is not None else TokenRecycle(config.tree)
   
    def reset(self):
        self.tree_model.reset()

    def lookup(self, start_token: int):
        return CandidateType.tree, self.tree_model.lookup(start_token)
    
    @profile_decorator("draft.update")
    def update(self, tokens: List[int], topk_nest: List[List[int]]):
        self.tree_model.update(tokens, topk_nest)
