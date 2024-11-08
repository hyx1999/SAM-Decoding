from .tree import TreeModel
from .token_recycle import TokenRecycle
from .eagle import Eagle
from typing import Dict, Union

tree_model_cls: Dict[
    str, 
    Union[TokenRecycle, Eagle]
] = {
    "token_recycle": TokenRecycle,
    "eagle": Eagle
}
