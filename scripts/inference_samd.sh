#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

# vicuna-7b-v1.3
CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd \
    --model-type vicuna \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-samd-eagle2 \
    --sam_path local_cache/sam_alpaca_vicuna-7b-v1.3_min-endpos.pkl \
    --tree_method eagle2 \
    --samd_n_predicts 40 \
    --samd_len_threshold 5 \
    --samd_len_bias 5 
