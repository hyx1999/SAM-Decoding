#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

# vicuna
CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_pld \
    --model-type vicuna \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-pld

# llama3
CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_pld \
    --model-type llama3 \
    --bench-name spec_bench \
    --model-path /home/wangke/models/Meta-Llama-3-8B-Instruct \
    --model-id llama3-8b-instruct-pld
