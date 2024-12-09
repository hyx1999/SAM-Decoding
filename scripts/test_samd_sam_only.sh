#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

CUDA_VISIBLE_DEVICES=${devices} \
    python -m tests.test_samd_sam_only \
    --sam_path local_cache/sam_alpaca_vicuna-7b-v1.3.pkl \
    --model_path /data/models/vicuna-7b-v1.3 \
    --device "cuda"
