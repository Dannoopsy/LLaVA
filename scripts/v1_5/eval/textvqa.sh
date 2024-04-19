#!/bin/bash

python3.11 -m llava.eval.model_vqa_loader \
    --model-path ../checkpoints/llava_gemma_lorarus \
    --model-base ../checkpoints/gemma-2b-it \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode gemma

python3.11 -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
