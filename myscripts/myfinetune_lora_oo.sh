# +
# #!/bin/bash

deepspeed ../../LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 1 --lora_alpha 2 --mm_projector_lr 2e-5 \
    --deepspeed ../../LLaVA/scripts/zero3.json \
    --model_name_or_path Open-Orca/oo-phi-1_5 \
    --version v1 \
    --data_path ../../data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ../../data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-base-patch16 \
    --pretrain_mm_mlp_adapter ../../checkpoints/llava_oophi_pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ../../checkpoints/llava_oophi_lora_extratiny \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
