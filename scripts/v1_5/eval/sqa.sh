#!/bin/bash

python3.11 -m llava.eval.model_vqa_science \
    --model-path ../checkpoints/llava_tinyllama \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python3.11 llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json
