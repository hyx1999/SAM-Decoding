#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

# vicuna-7b-v1.3
CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_sam_only \
    --model-type vicuna \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-samd_sam_only \
    --sam_path local_cache/sam_alpaca_vicuna-7b-v1.3.pkl \
    --samd_max_predicts 60 \
    --samd_alpha 4.0 \
    --samd_len_bias 0
