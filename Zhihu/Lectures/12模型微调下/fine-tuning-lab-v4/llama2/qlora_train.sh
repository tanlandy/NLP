#! /usr/bin/env bash

set -ex

LR=2e-4
LORA_RANK=8

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=hotel_qlora
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

DATA_FS="/opt/models"

# Sets a default value for the local rank in distributed training. -1 usually means it's not set or used.
LOCAL_RANK=-1 

# only use GPU 0 for training and evaluation.
CUDA_VISIBLE_DEVICES=0 

python main_qlora.py \
    --do_train \
    --do_eval \
    --train_file ../data/train.llama2.jsonl \
    --validation_file ../data/dev.llama2.jsonl \
    --prompt_column context \  # Specifies the column name in the dataset that contains the prompts.
    --response_column response \
    --overwrite_cache \  # Indicates to overwrite any cached data.
    --model_name_or_path "${DATA_FS}/Llama-2-7b-hf" \
    --output_dir $OUTPUT_DIR \
    --optim "paged_adamw_8bit" \
    --max_source_length 2048 \  # Maximum number of tokens in the source text.
    --max_target_length 1024 \
    --per_device_train_batch_size 1 \  # Batch size per device during training.
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \  # Number of steps to accumulate gradients before updating model weights.
    --evaluation_strategy steps \
    --eval_steps 300 \  # Perform an evaluation every 300 steps.
    --num_train_epochs 1 \  # Number of training epochs.
    --logging_steps 300 \  # Log metrics every 300 steps.
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps 300 \  # Save the model every 300 steps.
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --fp16 \  # Use 16-bit floating-point precision (half precision) for training.
    --warmup_ratio 0.1 \  # Ratio of training to perform linear learning rate warmup.
    --seed 23 2>&1 | tee ${OUTPUT_DIR}/train.log  # Redirects all output (stdout and stderr) to a log file while also displaying it on the terminal.
