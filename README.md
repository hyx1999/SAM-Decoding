# SAM-Decoding

<p align="center">
| <a href="https://arxiv.org/abs/2411.10666"><b>Paper</b></a> |
</p>

---

*News* 🔥
- [2024/11] SAM-Decoding is now available on [arXiv](https://arxiv.org/abs/2411.10666).

---

## Introduction

SAM-Decoding introduces a new speculative decoding technique designed for Large Language Models (LLMs). This method is particularly suited for scenarios where the model's generated content overlaps with the input context or existing textual information. It is primarily aimed at applications where the model's output aligns with the given prompt or text base, such as summarization, retrieval-augmented generation, code editing, and document-based question answering. Moreover, SAM-Decoding maintains performance levels that are comparable to those of the leading speculative decoding methods in other domains.

Our work can be seen as a possible open source implementation of the [OpenAI Predicted Outputs](https://community.openai.com/t/introducing-predicted-outputs/1004502) feature.

## Method

The idea behind SAM Decoding is to retrieve possible model outputs from a prompt or knowledge base, thereby reducing the number of model inferences.

Although some existing works is also based on this idea

- [PLD - prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding): matching n-grams from the prompt in a brute-force based manner.

- [REST - Retrieval-Based Speculative Decoding](https://github.com/FasterDecoding/REST): matching n-grams from a text base using suffix arrays.

They have two main limitations:

1. The n-gram matching method they use limits the accuracy of the retrieval results and is also not efficient enough.

2. They perform well in scenarios such as summarization, but have almost no acceleration effect in other domains such as translation.

In contrast, SAM-Decoding uses [suffix automaton](https://en.wikipedia.org/wiki/Suffix_automaton) to generate output by finding the longest suffix match of the current generated text from the prompt and the text base. Thanks to the excellent properties of the suffix automaton, the average time complexity of finding the longest suffix match is O(1), which is faster and more accurate than PLD and REST.

At the same time, since SAM-Decoding can compute the longest matching length, it can be combined with the draft model based speculative decoding methods, such as [EAGLE](https://github.com/SafeAILab/EAGLE) and [Token Recycle](https://arxiv.org/abs/2408.08696) that is, automatically selecting to use the retrieval result as the draft or the generation result of the draft model as the draft according to the suffix matching length.

## Experiment

Expeiment result on [Spec-Bench](https://github.com/hemingkx/Spec-Bench)

**warning: Please note that these results are not final and may be revised**

- Device: a single NVIDIA RTX A6000 GPU (48GB) with 20 CPU cores
- Testing environment: Pytorch 2.3.0, under CUDA 12.1
- Experimental Settings: Vicuna-7B-v1.3, greedy decoding, FP16 precision, batch size = 1

| Models                                                        | Multi-turn Conversation | Translation | Summa-rization | Question Answering | Mathematical Reasoning | Retrieval-aug. Generation | #Mean Accepted Tokens |  Overall  |
| ------------------------------------------------------------  | :---------------------: | :---------: | :------------: | :----------------: | :--------------------: | :-----------------------: | :-------------------: | :-------: |
| SAM-Decoding\[EAGLE2\]                                        |          3.02x          |    1.89x    |     2.76x      |       2.19x        |         2.83x          |           2.23x           |         4.61          |   2.49x   |
| [EAGLE2](https://github.com/SafeAILab/EAGLE)                  |          2.87x          |    1.92x    |     2.33x      |       2.20x        |         2.88x          |           2.03x           |         4.36          |   2.38x   |
| SAM-Decoding\[EAGLE\]                                         |          2.78x          |    1.88x    |     2.65x      |       2.12x        |         2.57x          |           2.10x           |         3.77          |   2.35x   |
| [EAGLE](https://huggingface.co/blog/assisted-generation)      |          2.63x          |    1.92x    |     2.28x      |       2.11x        |         2.64x          |           1.95x           |         3.57          |   2.27x   |
| SAM-Decoding\[Token Recycle\]                                 |          2.48x          |    1.73x    |     2.86x      |       1.98x        |         2.44x          |           2.14x           |         3.03          |   2.27x   |
| [Tokey Recycle](https://arxiv.org/abs/2408.08696)             |          1.92x          |    1.61x    |     1.96x      |       1.71x        |         2.16x          |           1.68x           |         2.83          |   1.84x   |

## Data

The test data for SAM-Decoding is available in the Spec-Bench repository. To proceed, you should place the relevant data files `Spec-Bench/data` from the [Spec-Bench](https://github.com/hemingkx/Spec-Bench) repository into the evaluation directory of our project `evaluation/data`.

## Inference

An example of using SAM-Decidubg is provided in `tests/test_samd.py`, which can be executed via `scripts/test_samd.sh`. 

Note that this script relies on a SAM (StaticSAM) built from a text base. We can build this SAM by executing the scripts `tools/gen_sam_none.py`.

In the experiment, we used SAM based on alpaca-clean, gsm8k, and python-instruction datasets. We can build this SAM by executing the scripts `tools/prepare_prompts.py`, `tools/gen_response.py`, `tools/gen_sam_alpaca.py` in sequence.

```python
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
    DraftModel,
    load_sam
)
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--samd_n_predicts', type=int, default=15)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument("--tree_method", type=str, default="token_recycle")
    parser.add_argument("--tree_model_path", type=str, default=None)
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'])
    args = parser.parse_args()
    args.dtype = {
        'float16': torch.float16,
        'float32': torch.float32,
    }[args.dtype]
    return args

@torch.inference_mode()
def samd_generate(args, inputs, model, tokenizer):
    sam = load_sam(args.sam_path)
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_model_path=args.tree_model_path,
    )
    draft = DraftModel(
        samd_config, 
        lm=model,
        dtype=args.dtype,
        device=args.device
    )
    samd_model = SamdModel(
        samd_config, 
        model, 
        draft, 
        tokenizer.eos_token_id,
        args.dtype,
        args.device,
    )
    samd_model.eval()
    
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
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
    )
    model.eval()
    
    prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n\nUSER: Embrace the role of Sheldon from \"The Big Bang Theory\" as we delve into our conversation. Don\u2019t start with phrases like \"As Sheldon\". Let's kick things off with the following question: \"What is your opinion on hand dryers?\"\n\nASSISTANT: "]

    inputs = tokenizer(
        prompts, 
        padding=True, 
        return_tensors="pt"
    ).to(args.device)
    
    samd_generate(args, inputs, model, tokenizer)

if __name__ == '__main__':
    main()

```