#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=2

# vicuna
CUDA_VISIBLE_DEVICES=${devices} python -m evaluation.inference_eagle \
    --model-type vicuna \
    --ea-model-path /data/models/EAGLE-Vicuna-7B-v1.3 \
    --base-model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-eagle \
    --bench-name spec_bench \
    --temperature 0 \
    --dtype "float16"

# vicuna
CUDA_VISIBLE_DEVICES=${devices} python -m evaluation.inference_eagle \
    --model-type llama3 \
    --ea-model-path /home/wangke/models/EAGLE-LLaMA3-Instruct-8B \
    --base-model-path /home/wangke/models/Meta-Llama-3-8B-Instruct \
    --model-id llama3-8b-instruct-eagle \
    --bench-name spec_bench \
    --temperature 0 \
    --dtype "float16"
