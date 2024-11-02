#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=2

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.speed \
        --file-path evaluation/data/spec_bench/model_answer/vicuna-7b-v1.3-sam_alpaca-v0.3.1.jsonl
