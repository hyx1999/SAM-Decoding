#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=2

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.equal \
    --jsonfile1 vicuna-7b-v1.3.jsonl --jsonfile2 vicuna-7b-v1.3-sam_alpaca.jsonl

    