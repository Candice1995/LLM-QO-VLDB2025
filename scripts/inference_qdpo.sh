#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Error: train-run-name is expected."
    echo "Usage: $0 <TRAIN_RUN_NAME>"
    exit 1
fi
TRAIN_RUN_NAME=$1

python QDPO.py \
    --dataset_dir ../data \
    --train_dataset_name "meta_final.dsb.dpo.train.one_shot_CoT.jsonl" \
    --test_dataset_name "meta_final.dsb.mixed.valid.one_shot_CoT.jsonl" \
    --valid_dataset_name "meta_final.dsb.dpo.valid.one_shot_CoT.jsonl" \
    --max_new_tokens 512 \
    --max_steps 200 \
    --save_steps 50 \
    --predict_dir predicts_dsb \
    --output_dir outputs_dsb \
    --llm_name llama3-8b \
    --eval_mode true \
    --train_run_name $TRAIN_RUN_NAME