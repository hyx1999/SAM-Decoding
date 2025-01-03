#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

# vicuna-7b-v1.3
CUDA_VISIBLE_DEVICES=${devices} python -m evaluation.inference_eagle \
    --model-type vicuna \
    --ea-model-path /data/models/EAGLE-Vicuna-7B-v1.3 \
    --base-model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-eagle \
    --bench-name spec_bench \
    --temperature 0 