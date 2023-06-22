#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$ACTOR_MODEL_PATH" == "" ]; then
    ACTOR_MODEL_PATH=/mnt/application/leyf/llm_zoo/bloom3b_yj/bloom-3B/
fi
if [ "$CRITIC_MODEL_PATH" == "" ]; then
    CRITIC_MODEL_PATH=/mnt/application/leyf/ds_chat/output/
fi
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/mnt/application/leyf/ds_chat/rlhf_output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 8 main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
   --num_padding_at_beginning 0 --gradient_accumulation_steps 2 \
   --deepspeed --actor_lora_dim 0   --disable_actor_dropout \
   --output_dir $OUTPUT --data_path /mnt/application/leyf/ds_chat/data/chinese_chatgpt_corpus \
   #&> $OUTPUT/training.log --actor_gradient_checkpointing
