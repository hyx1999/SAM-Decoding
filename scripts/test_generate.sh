#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=2

CUDA_VISIBLE_DEVICES=${devices} python main.py \
    --sam_path local_cache/sam_mini.pkl \
    --model_path /data/models/vicuna-7b-v1.3 \
    --device "cuda"
