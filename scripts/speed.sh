#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=2

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.speed
    