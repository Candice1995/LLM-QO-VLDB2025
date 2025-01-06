#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Error: checkpoint path is expected."
    echo "Usage: $0 <CKPT_PATH>"
    exit 1
fi
CKPT_PATH=$1

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
    --ckpt_name $CKPT_PATH \
    --eval_mode false \