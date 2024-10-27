import os
import argparse
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from .data_utils import process_alpaca, process_gsm8k, process_vocab

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/data/models/vicuna-7b-v1.3')
parser.add_argument('--cutoff_len', type=int, default=1024)
parser.add_argument('--prompt_template_name', type=str, default='vicuna')
args = parser.parse_args()

model_name = args.model_name.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

alpaca_data = load_dataset('json', data_files='sam_data/alpaca-cleaned/alpaca_data_cleaned.json', split="train")
code_data = load_dataset('parquet', data_files='sam_data/python_code_instructions_18k_alpaca/data/*.parquet', split="train")
math_data = load_dataset('parquet', data_files='sam_data/gsm8k/main/train-*.parquet', split="train")

alpaca_data = process_alpaca(args, alpaca_data)
code_data = process_alpaca(args, code_data)
math_data = process_gsm8k(args, math_data)
# vocab_data = process_vocab(args, tokenizer)

sam_data: Dataset = concatenate_datasets([alpaca_data, code_data, math_data])
sam_data.save_to_disk(f'sam_data/sam_prompts')

print(sam_data)
print(sam_data[:10])
