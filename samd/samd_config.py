from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal
from enum import Enum

@dataclass
class SamdConfig:
    n_predicts: int = field(default=10)
    len_threshold: int = field(default=4)
    len_bias: int = field(default=4)
    tree: Optional[List[List[int]]] = field(default=None)

    def __post_init__(self):
        if self.tree is None:
            self.tree = load_default_tree()


class ForwardType(str, Enum):
    prefill = "prefill"
    seq_decode = "seq_decode"
    tree_decode = "tree_decode"

class ForwardState:
        
    def __init__(self, forward_type: ForwardType | None) -> None:
        self.forward_type = forward_type


def load_default_tree():
    import os
    import json
    samd_path = os.path.dirname(__file__)
    with open(os.path.join(samd_path, "config", "token_recycle.json"), "r") as f:
        tree_dict: dict = json.load(f)
    num_node = len(tree_dict)
    tree: List[List[int]] = []
    for i in range(num_node):
        tree.append(tree_dict[str(i)])
    return tree
