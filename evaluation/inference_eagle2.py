"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import torch
import argparse
from fastchat.utils import str_to_torch_dtype

from evaluation.eval import run_evals, reorg_answer_files
from evaluation.model.eagle2.ea_model import EaModel
from functools import partial

def ea_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, is_llama3=False):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids, new_token, step, accept_length_list = model.eagenerate(
        torch.as_tensor(input_ids).cuda(),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        log=True,
        is_llama3=is_llama3,
    )
    return input_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=str,
        default="vicuna",
        choices=["vicuna", "llama3"]
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["vicuna", "llama3"]
    )
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/70B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="ess-vicuna-70b-fp16")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument("--is_llama3", action="store_true")

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)

    question_file = f"evaluation/data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"evaluation/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    if args.model_type == "llama3":
        ea_forward = partial(ea_forward, is_llama3=True)

    run_evals[args.template](
        model=model,
        tokenizer=tokenizer,
        forward_func=ea_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
    )

    reorg_answer_files[args.template](answer_file)
