#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/mnt/application/leyf/ds_chat/output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py --data_path /mnt/application/leyf/ds_chat/data/chinese_chatgpt_corpus --model_name_or_path /mnt/application/leyf/llm_zoo/bloomz560m \
   --num_padding_at_beginning 0 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
