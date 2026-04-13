#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ZERO_STAGE=$1
OUTPUT=./output_llama2_7b_lora
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

# LoRA + ZeRO-3 fits 2x 16GB. Prefer bf16: no FP16 loss-scaler (avoids
# "loss scale already at minimum"). Use fp16 + lower --lora_learning_rate if
# your GPU lacks bf16. Add --offload if you still OOM.
deepspeed main.py \
   --data_split 2,4,4 \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --dtype bf16 \
   --compute_fp32_loss \
   --lora_dim 16 \
   --lora_module_name "model.layers." \
   --only_optimize_lora \
   --lora_learning_rate 1e-4 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
