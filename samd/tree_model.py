import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm

from .samd_config import SamdConfig

class TreeModel:
    
    def __init__(self,
        tree: List[List[int]]
    ) -> None:
        self.tree = tree
        self.cache = {}
        
    def reset(self):
        # do nothting
        pass
    
    def update(self, tokens: List[int], topk_nest: List[List[int]]):
        for token, next_token, topk in zip(tokens, tokens[1:] + [None], topk_nest):
            if next_token != topk[0] and next_token is not None:
                topk = [next_token] + topk[:-1]
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
