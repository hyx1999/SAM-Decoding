import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from tqdm import tqdm


class Tree:
    
    def reset(self):
        raise NotImplementedError
    
    def update(self, tokens: List[int], topk_nest: List[List[int]]):
        raise NotImplementedError
    
    def lookup(self, start_token: int) -> List[int]:
        raise NotImplementedError
