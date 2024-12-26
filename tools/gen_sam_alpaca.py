import os
import argparse
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from samd import SamdConfig, build_sam, dump_sam

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/data/models/vicuna-7b-v1.3')
parser.add_argument('--sam_data_path', type=str, default='sam_data/sam_dialogues')
parser.add_argument('--cutoff_len', type=int, default=2048)
parser.add_argument('--n_predicts', type=int, default=10)
parser.add_argument('--sam_path', type=str, default="local_cache/sam_alpaca_vicuna-7b-v1.3_min-endpos.pkl")
args = parser.parse_args()

sam_data = load_from_disk(args.sam_data_path)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def tokenize_fn(data_point, add_eos_token=False):
    text = data_point["prompt"] + data_point["response"]
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        text,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < args.cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    return result

column_names = sam_data.column_names

batch_tokens = sam_data.map(
    tokenize_fn,
    desc=f"Processing sam dialogue datasets",
)["input_ids"]
for i in range(len(tokenizer)):
    batch_tokens.append([i])

sam = build_sam(batch_tokens, tokenizer.eos_token_id)

model_name = args.model_name.split("/")[-1]
dump_sam(args.sam_path, sam)
