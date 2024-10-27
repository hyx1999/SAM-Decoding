import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from samd import (
    SamdConfig, 
    SamdModel, 
    GenerationConfig, 
    load_sam
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_path', type=str, required=True)
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

    sam = load_sam(args.sam_path)

    samd_config = SamdConfig(n_gram=args.samd_n_gram, k=args.samd_k)
    samd_model = SamdModel(samd_config, model, sam, tokenizer.eos_token_id)
    
    assert sam.n_gram == samd_config.n_gram
    assert sam.k == samd_config.k
    
    prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Give three tips for staying healthy.\n\nASSISTANT:"]

    inputs = tokenizer(
        prompts, 
        padding=True, 
        return_tensors="pt"
    ).to("cuda")
    
    tokens = samd_model.generate(**inputs)
    response = tokenizer.batch_decode(tokens)
    print("response:\n{}".format(response))


if __name__ == '__main__':
    main()
