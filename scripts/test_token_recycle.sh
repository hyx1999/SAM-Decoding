#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

CUDA_VISIBLE_DEVICES=${devices} \
    python -m tests.test_token_recycle \
    --model_path /data/models/vicuna-7b-v1.3 \
    --device "cuda"
