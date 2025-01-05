#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Error: train-run-name is expected."
    echo "Usage: $0 <TRAIN_RUN_NAME>"
    exit 1
fi
TRAIN_RUN_NAME=$1

python QIT.py \
        --dataset_dir ../data  \
        --train_dataset_name "meta_final.dsb.mixed.train.one_shot_CoT.jsonl" \
        --test_dataset_name meta_final.dsb.mixed.valid.one_shot_CoT.jsonl \
        --valid_dataset_name meta_final.dsb.mixed.valid.one_shot_CoT.jsonl \
        --max_steps 600 \
        --save_steps 100 \
        --max_new_tokens 512 \
        --predict_dir predicts_dsb \
        --output_dir outputs_dsb \
        --llm_name llama3-8b \
        --eval_mode true \
        --train_run_name $TRAIN_RUN_NAME