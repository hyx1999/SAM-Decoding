#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=2

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-sam_alpaca-v0.5.1 \
    --sam_path local_cache/sam_alpaca.pkl \
    --tree_method eagle2 \
    --samd_n_predicts 15 \
    --samd_len_threshold 5 \
    --samd_len_bias 5 
