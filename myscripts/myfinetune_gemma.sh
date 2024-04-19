#!/bin/bash

deepspeed ../../LLaVA/llava/train/train_mem.py \
    --deepspeed ../../LLaVA/scripts/zero3.json \
    --model_name_or_path ../../checkpoints/gemma-2b-it \
    --version gemma \
    --data_path ../../data/finetunedata/corrected_data.json \
    --val_path ../../data/vqav2val/dataval80537.json \
    --image_folder ../../data/finetunedata \
    --val_image_folder ../../data/vqav2val \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ../../checkpoints/llava_gemma_pretrained/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ../../checkpoints/llava_gemma_ft \
    --num_train_epochs 2 \
    --eval_steps 0.24 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --eval_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb