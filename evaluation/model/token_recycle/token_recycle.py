import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm

from .token_recycle_config import TokenRecycleConfig

class TokenRecycle:
    
    def __init__(self,
        tree: List[List[int]]
    ) -> None:
        self.tree = tree
        self.cache = {}
        
    def reset(self):
        pass  # do nothting
    
    def update(self, tokens: List[int], topk_nest: List[List[int]]):
        for token, topk in zip(tokens, topk_nest):
            self.cache[token] = topk
    
    def lookup(self, start_token: int) -> List[int]:
        tree_tokens = [start_token] + [0] * (len(self.tree) - 1)
        for node_id, childs in enumerate(self.tree):
            token = tree_tokens[node_id]
            if token not in self.cache:
                continue
            topk = self.cache[token]
            for child_id, child in enumerate(childs):
                tree_tokens[child] = topk[child_id]
        return tree_tokens
