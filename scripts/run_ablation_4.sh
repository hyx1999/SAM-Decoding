#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=3

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd \
    --model-type vicuna \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-sam_alpaca-abl4 \
    --sam_path local_cache/sam_none.pkl \
    --tree_method token_recycle \
    --samd_n_predicts 15 \
    --samd_len_threshold 0 \
    --samd_len_bias 5 \
    --samd_tree_path default_tree_1_1.json
