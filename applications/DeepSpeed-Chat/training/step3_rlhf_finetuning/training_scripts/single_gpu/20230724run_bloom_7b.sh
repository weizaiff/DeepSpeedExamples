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
    ACTOR_MODEL_PATH=/mnt/application/leyf/llm_zoo/mmm/output/20230606bloom7b1-duojiduoka
fi
if [ "$CRITIC_MODEL_PATH" == "" ]; then
    CRITIC_MODEL_PATH=/data/application/leyf/ds_chat/20230703_step2_output
fi
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/mnt/application/leyf/ds_chat/rlhf_output/20230724__1ppepoch_add_specialtoken_bloom7b1_from_sft
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed  --include "localhost:1,2,3,4,5,6,7" --master_port 29501 main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
   --num_padding_at_beginning 0 --gradient_accumulation_steps 4 \
   --deepspeed --actor_lora_dim 0   --disable_actor_dropout \
   --output_dir $OUTPUT --data_path /mnt/application/leyf/llm_zoo/mmm/data/rlhf_step3 \
   --ppo_epochs 1 --num_train_epochs 1 \
   &>> $OUTPUT/training.log #--actor_gradient_checkpointing
