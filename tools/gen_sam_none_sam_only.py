import os
import argparse
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from samd_sam_only import SamdConfig, build_sam, dump_sam

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/data/models/vicuna-7b-v1.3')
parser.add_argument('--cutoff_len', type=int, default=2048)
parser.add_argument('--n_predicts', type=int, default=10)
parser.add_argument('--sam_path', type=str, default="local_cache/sam_none.pkl")
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_name)

batch_tokens = []
for i in range(len(tokenizer)):
    batch_tokens.append([i])

sam = build_sam(batch_tokens, tokenizer.eos_token_id)

model_name = args.model_name.split("/")[-1]
dump_sam(args.sam_path, sam)
