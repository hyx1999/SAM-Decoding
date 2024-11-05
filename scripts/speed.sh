#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

python -m evaluation.speed \
    --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-sam_alpaca-v0.3.4.3.jsonl

# python -m evaluation.speed \
#     --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-sam_none-v0.3.4.jsonl
