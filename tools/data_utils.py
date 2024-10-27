import os
import argparse
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from accelerate.logging import get_logger
from itertools import chain
from .prompter import Prompter

logger = get_logger(__name__)

def process_gsm8k(
    args,
    raw_dataset: Dataset
):
    column_names = raw_dataset.column_names
    prompter = Prompter(args.prompt_template_name)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["question"],
        )
        return {"prompt": full_prompt}

    lm_dataset = raw_dataset.map(
        generate_and_tokenize_prompt,
        remove_columns=column_names,
        desc=f"Processing gsm8k datasets",
    )

    return lm_dataset

def process_alpaca(
    args,
    raw_dataset: Dataset
):
    column_names = raw_dataset.column_names
    prompter = Prompter(args.prompt_template_name)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
        )
        return {"prompt": full_prompt}

    lm_dataset = raw_dataset.map(
        generate_and_tokenize_prompt,
        remove_columns=column_names,
        desc=f"Processing alpaca like datasets",
    )

    return lm_dataset
