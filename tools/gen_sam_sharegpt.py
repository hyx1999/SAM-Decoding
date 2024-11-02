import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from samd import SamdConfig, build_sam, dump_sam

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/data/models/vicuna-7b-v1.3')
parser.add_argument('--sam_data_path', type=str, default='/data/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json')
parser.add_argument('--cutoff_len', type=int, default=2048)
parser.add_argument('--n_gram', type=int, default=8)
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--sam_path', type=str, default="local_cache/sam_sharegpt.pkl")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

dataset_path = args.sam_data_path
batch_tokens = []
dataset = json.load(open(dataset_path))
total_length = len(dataset)
print("number of samples: ", total_length)
for conversations in tqdm(dataset, total=total_length):
    for sample in conversations['conversations']:
            token_list = tokenizer.encode(
                sample['value'], 
                truncation=True,
                max_length=args.cutoff_len, 
            )
            batch_tokens.append(token_list)

samd_config = SamdConfig(n_gram=args.n_gram, k=args.k)

sam = build_sam(samd_config, batch_tokens, tokenizer.eos_token_id)

model_name = args.model_name.split("/")[-1]
dump_sam(args.sam_path, sam)
