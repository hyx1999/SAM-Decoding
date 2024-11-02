#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=1

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd \
    --bench-name spec_bench \
    --model-path /data/models/vicuna-7b-v1.3 \
    --model-id vicuna-7b-v1.3-sam_alpaca-v0.3.1 \
    --sam_path local_cache/sam_alpaca.pkl \
    --samd_k 4
