#! /usr/bin/env bash
# 一般LR从2e-4开始，然后根据实际情况调整
# lora_alpha常用是32，64
# batch_size可以自己调整，尝试出来2的效果最好

set -ex

LR=2e-4
MAX_SEQ_LEN=3072

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=hotel_lora
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

BASE_MODEL_PATH=/root/autodl-tmp/chatglm3-6b

CUDA_VISIBLE_DEVICES=0 python main_lora.py \
    --do_train \
    --do_eval \
    --train_file ../data/train.chatglm3.jsonl \
    --validation_file ../data/dev.chatglm3.jsonl \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 4 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --logging_steps 1 \
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps 300 \
    --learning_rate $LR \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 2>&1 | tee ${OUTPUT_DIR}/train.log
