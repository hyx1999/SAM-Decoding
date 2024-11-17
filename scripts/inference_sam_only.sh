#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=3

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_sam_only \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-sam_only \
    --samd_n_predicts 15 \
    --samd_len_bias 5 

# --sam_path local_cache/sam_alpaca.pkl \
