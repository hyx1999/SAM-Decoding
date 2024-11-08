#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

python -m evaluation.speed \
    --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-sam_alpaca-v0.5.1.1.jsonl

# python -m evaluation.speed \
#     --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-pld.jsonl

# python -m evaluation.speed \
#     --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-token_recycle.jsonl

# python -m evaluation.speed \
#     --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-eagle-temperature-0.0.jsonl

# python -m evaluation.speed \
#     --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-eagle2-temperature-0.0.jsonl
