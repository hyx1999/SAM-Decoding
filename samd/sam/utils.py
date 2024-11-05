import time
import pickle
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from typing import List

from .sam import StaticSAM
from ..samd_config import SamdConfig

def build_sam(
    config: SamdConfig,
    batch_tokens: List[List[int]],
    eos_token: int,
):
    sam = StaticSAM.build(
        batch_tokens, 
        eos_token, 
        config.n_predicts
    )
    return sam

def dump_sam(path: str, sam: StaticSAM):
    with open(path, "wb") as f:
        pickle.dump(sam, f)

def load_sam(path: str):
    print("load sam...")
    start = time.perf_counter()
    with open(path, "rb") as f:
        sam: StaticSAM = pickle.load(f)
    end = time.perf_counter()
    assert type(sam) is StaticSAM
    print("loading ended in {} seconds.".format(end - start))
    return sam
