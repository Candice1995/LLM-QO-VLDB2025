#!/bin/bash
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
        --eval_mode false \