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
from evaluation.model.token_recycle import (
    TokenRecycleConfig, 
    TokenRecycleModel, 
    TokenRecycleGenerationConfig,
    DraftModel
)
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--samd_n_predicts', type=int, default=15)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_cache_len', type=int, default=2048)
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'])
    args = parser.parse_args()
    args.dtype = {
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]
    return args

@torch.inference_mode()
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


@torch.inference_mode()
def token_recycle_generate(args, inputs, model, tokenizer):
    assert inputs.input_ids.shape[-1] + args.max_new_tokens <= args.max_cache_len
    token_recycle_config = TokenRecycleConfig(n_predicts=args.samd_n_predicts)
    draft = DraftModel(token_recycle_config)
    token_recycle_model = TokenRecycleModel(
        token_recycle_config, 
        model, 
        draft, 
        tokenizer.eos_token_id,
        args.dtype,
        args.device,
    )
    
    gen_config = TokenRecycleGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
    )

    st = time.perf_counter()
    outputs = token_recycle_model.generate(**inputs, generation_config=gen_config)
    ed = time.perf_counter()
    response = tokenizer.decode(outputs.output_ids[0])
    print("model inference time use: {} seconds".format(ed - st))
    print("token_recycle_model response:\n{}".format(repr(response)))
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
    )
    model.eval()
    
    prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Give three tips for staying healthy.\n\nASSISTANT: "]
    
    # prompts = ['A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n\nUSER: Please generate the following: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10".\n\nASSISTANT: ']
    
    # prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n\nUSER: Embrace the role of Sheldon from \"The Big Bang Theory\" as we delve into our conversation. Don\u2019t start with phrases like \"As Sheldon\". Let's kick things off with the following question: \"What is your opinion on hand dryers?\"\n\nASSISTANT: "]

    inputs = tokenizer(
        prompts,
        padding=True, 
        return_tensors="pt"
    ).to(args.device)
    
    generate(args, inputs, model, tokenizer)

    token_recycle_generate(args, inputs, model, tokenizer)

if __name__ == '__main__':
    main()
