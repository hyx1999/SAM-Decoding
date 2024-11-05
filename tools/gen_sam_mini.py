import os
import argparse
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from samd import SamdConfig, build_sam, dump_sam

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/data/models/vicuna-7b-v1.3')
parser.add_argument('--cutoff_len', type=int, default=2048)
parser.add_argument('--n_predicts', type=int, default=10)
parser.add_argument('--sam_path', type=str, default="local_cache/sam_mini.pkl")
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_name)

prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Give three tips for staying healthy.\n\nASSISTANT: 1. Eat a balanced diet that includes a variety of fruits, vegetables, whole grains, and lean proteins.\n2. Get regular exercise, such as walking, jogging, or lifting weights.\n3. Get enough sleep and take breaks throughout the day to rest and recharge."

token_ids = tokenizer(prompt).input_ids


batch_tokens = []
batch_tokens.append(token_ids)
for i in range(len(tokenizer)):
    batch_tokens.append([i])

samd_config = SamdConfig(n_predicts=args.n_predicts)

sam = build_sam(samd_config, batch_tokens, tokenizer.eos_token_id)

model_name = args.model_name.split("/")[-1]
dump_sam(args.sam_path, sam)
