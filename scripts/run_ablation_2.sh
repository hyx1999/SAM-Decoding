#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=2

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-sam_alpaca-abl2 \
    --sam_path local_cache/sam_alpaca.pkl \
    --tree_method token_recycle \
    --samd_n_predicts 40 \
    --samd_len_threshold 100010 \
    --samd_len_bias -100000 \
    --samd_tree_path default_tree_6_60.json
