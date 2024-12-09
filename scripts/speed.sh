#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

python -m evaluation.speed \
    --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-samd_sam_only.jsonl

python -m evaluation.speed \
    --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-samd-token_recycle.jsonl

python -m evaluation.speed \
    --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-samd-eagle2.jsonl
