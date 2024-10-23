import pickle
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from .sam import SAM
from ..samd_config import SamdConfig

def build_sam(
    config: SamdConfig,
    dataset: Dataset, 
    tokenizer: PreTrainedTokenizerFast, 
    column_name: str | None = None
):
    if column_name is None:
        column_name = dataset.column_names[0]
    def tokenize_fn(examples):
        return tokenizer(examples[column_name])
    tokenized_datasets = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    sam = SAM.build(
        tokenized_datasets["input_ids"], 
        tokenizer.eos_token_id, 
        config.n_gram,
        config.k,
    )
    return sam

def dump_sam(path: str, sam: SAM):
    with open(path, "wb") as f:
        pickle.dump(sam, f)

def load_sam(path: str):
    with open(path, "rb") as f:
        sam: SAM = pickle.load(f)
    assert type(sam) is SAM
    return sam
