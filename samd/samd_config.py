from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal
from enum import Enum

@dataclass
class SamdConfig:
    n_gram: int = field(default=8)
    k: int = field(default=8)
    tree: Optional[List[List[int]]] = field(default=None)

    def __post_init__(self):
        if self.tree is None:
            self.tree = load_default_tree()


class ForwardType(str, Enum):
    prefill = "prefill"
    decode = "decode"

class ForwardState:
        
    def __init__(self, forward_type: ForwardType | None) -> None:
        self.forward_type = forward_type


def load_default_tree():
    import os
    import json
    samd_path = os.path.dirname(__file__)
    with open(os.path.join(samd_path, "config", "default_tree.json"), "r") as f:
        tree_dict: dict = json.load(f)
    num_node = len(tree_dict)
    tree: List[List[int]] = []
    for i in range(num_node):
        tree.append(tree_dict[str(i)])
    return tree
