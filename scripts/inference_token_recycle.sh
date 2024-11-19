#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=2

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_token_recycle \
    --model-type vicuna \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-token_recycle

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_token_recycle \
    --model-type llama3 \
    --bench-name spec_bench \
    --model-path /home/wangke/models/Meta-Llama-3-8B-Instruct \
    --model-id llama3-8b-instruct-token_recycle
