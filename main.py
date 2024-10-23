import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from samd import (
    SamdConfig, 
    SamdModel, 
    GenerationConfig, 
    build_sam, load_sam, dump_sam
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_load_path', type=str, default=None)
    parser.add_argument('--sam_dump_path', type=str, default=None)
    parser.add_argument('--sam_dataset_path', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--samd_n_gram', type=int, default=16)
    parser.add_argument('--samd_k', type=int, default=8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda")

    samd_config = SamdConfig(n_gram=args.samd_n_gram, k=args.samd_k)
    
    if args.sam_load_path is not None:
        sam = load_sam(args.sam_load_path)
    elif args.sam_dataset_path is not None:
        sam_dataset = load_dataset(args.sam_dataset_path)
        sam = build_sam(samd_config, sam_dataset, tokenizer)
        if args.sam_dump_path is not None:
            dump_sam(args.sam_dump_path, sam)
    else:
        raise ValueError
    
    samd_model = SamdModel(samd_config, model, sam, tokenizer.eos_token_id)
    
    ...


if __name__ == '__main__':
    main()
