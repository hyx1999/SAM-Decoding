import argparse
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    GenerationMixin,
    LlamaConfig,
    LlamaTokenizer
)
from samd import (
    SamdConfig, 
    SamdModel, 
    SamdGenerationConfig,
    load_sam
)
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--samd_n_gram', type=int, default=8)
    parser.add_argument('--samd_k', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_cache_len', type=int, default=1024)
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'])
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
    args = parser.parse_args()
    args.dtype = {
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]
    return args


def generate(args, inputs, model, tokenizer):
    assert inputs.input_ids.shape[-1] + args.max_new_tokens <= args.max_cache_len
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, cache_implementation="static",
        cache_config = {
            "batch_size": 1,
            "max_cache_len": args.max_cache_len,            
        }
    )
    st = time.perf_counter()
    tokens = model.generate(**inputs, generation_config=gen_config)[0]
    ed = time.perf_counter()
    response = tokenizer.decode(tokens)
    print("model inference time use: {} seconds".format(ed - st))
    print("model response:\n{}".format(repr(response)))


def samd_generate(args, inputs, model, tokenizer):
    assert inputs.input_ids.shape[-1] + args.max_new_tokens <= args.max_cache_len
    sam = load_sam(args.sam_path)
    
    samd_config = SamdConfig(n_gram=args.samd_n_gram, k=args.samd_k)
    samd_model = SamdModel(
        samd_config, model, sam, 
        tokenizer.eos_token_id,
        args.device, 
        args.dtype,
    )
    
    assert sam.n_gram == samd_config.n_gram
    assert sam.k == samd_config.k
    
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
    )

    st = time.perf_counter()
    outputs = samd_model.generate(**inputs, generation_config=gen_config)
    ed = time.perf_counter()
    response = tokenizer.decode(outputs.output_ids[0])
    print("model inference time use: {} seconds".format(ed - st))
    print("samd_model response:\n{}".format(repr(response)))
    print("decode_steps: {}".format(outputs.decode_steps))
    print("decode_tokens: {}".format(outputs.decode_tokens))
    print("accepect_length_per_step: {}".format(outputs.accepet_length_per_step))

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=args.dtype, 
        device_map=args.device,
        # attn_implementation="eager",
    )
    
    prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Give three tips for staying healthy.\n\nASSISTANT: "]

    inputs = tokenizer(
        prompts, 
        padding=True, 
        return_tensors="pt"
    ).to(args.device)
    
    generate(args, inputs, model, tokenizer)

    samd_generate(args, inputs, model, tokenizer)

if __name__ == '__main__':
    main()
