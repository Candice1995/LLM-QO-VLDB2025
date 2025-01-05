# One must patch the DPO Trainer first!
#!/usr/bin/env python
# coding: utf-8
import argparse
from tqdm import tqdm
import numpy as np
import json
import os
from datetime import datetime

import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datetime import datetime
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import DPOTrainer
from unsloth import PatchDPOTrainer
PatchDPOTrainer()
import wandb, os
import os
import re
from typing import List, Literal, Optional
import pprint

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Set parameters.")
parser.add_argument('--dataset_dir', type=str, help='Path to the dataset directory.', default="/home/jtan/LLM-QO/data")
parser.add_argument('--train_dataset_name', type=str, help='Name of the training dataset.', default="train.jsonl")
parser.add_argument('--valid_dataset_name', type=str, help='Name of the validation dataset.', default="valid.jsonl")
parser.add_argument('--test_dataset_name', type=str, help='Name of the testing dataset.', default="test.jsonl")
parser.add_argument('--max_steps', type=int, help='Max number of training steps.', default=200)
parser.add_argument('--save_steps', type=int, help='Number of save steps .', default=50)
parser.add_argument('--llm_name', type=str, help='Name of the language model.', default="llama3-8b")
parser.add_argument('--ckpt_name', type=str, help='Name of the checkpoint of sft model. ', default="train_run_name")
parser.add_argument('--predict_dir', type=str, help='Name of predict directory.', default="predicts_dsb")
parser.add_argument('--output_dir', type=str, help='Name of output directory.', default="outputs_dsb")
parser.add_argument('--max_new_tokens', type=int, help='The maximum numbers of tokens to generate.', default=128)
parser.add_argument('--ckpt_step', type=str, help='The training step of sft model.', default="600")
parser.add_argument('--beta', type=float, help='beta.', default=0.1)
parser.add_argument('--eval_mode', type=str, help='Whether only to eval.', default="false")
parser.add_argument('--train_run_name', type=str, help='Name of the W&B run.', default="train_run_name")

args = parser.parse_args()


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

print(dtype)
print(load_in_4bit)



alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""


if args.eval_mode == "false":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"{args.ckpt_name}",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )



    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        chosens      = examples["chosen"]
        rejecteds      = examples["rejected"]
        text_prompts = []
        text_chosens = []
        text_rejecteds = []
        for instruction, input, chosen, rejected in zip(instructions, inputs, chosens, rejecteds):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input)
            text_prompts.append(text)
            text_chosens.append(chosen+ EOS_TOKEN)
            text_rejecteds.append(rejected+ EOS_TOKEN)
        return { "prompt" : text_prompts,"chosen":text_chosens, "rejected":text_rejecteds }
    pass


    raw_datasets = load_dataset(args.dataset_dir, data_files={"train": args.train_dataset_name, "valid": args.valid_dataset_name})


    column_names = list(raw_datasets["train"].features)

    raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42)  
    raw_datasets = raw_datasets.map(
        # apply_chat_template,
        formatting_prompts_func,
        num_proc = 12,
        remove_columns = column_names,
        batched=True,
    )



    row = raw_datasets["train"][0]
    pprint.pprint(raw_datasets["train"][8]["prompt"])
    shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)  
    raw_datasets["train"] = shuffled_train_dataset


    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["valid"]

    prompt_length = int(max(len(tokenizer(x)["input_ids"]) for x in train_dataset["prompt"]))
    max_seq_length_chosen = int(max(len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) for x in train_dataset))
    max_seq_length_rejected = int(max(len(tokenizer(x["prompt"] + x["rejected"])["input_ids"]) for x in train_dataset))
    max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)
    
    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(eval_dataset): {len(eval_dataset)}")
    
    prompt_length = ((prompt_length + 1) // 2) * 2
    max_seq_length = ((max_seq_length + 1) // 2) * 2
    print(f"max prompt length: {prompt_length}")
    print(f"max prompt + chosen length: {max_seq_length}")


    data_dir = args.dataset_dir
    data_dir = data_dir.split("/")[1]
    train_data_name = args.train_dataset_name.split(".jsonl")[0]

    run_name = "DPO_" + str(args.ckpt_step) +"_"+ str(args.beta) +"_" + data_dir + "_" + train_data_name + "_"+ args.llm_name + "-" 
    train_run_name = f"{run_name}-{datetime.now().strftime('%Y-%m-%d')}"

    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_ratio = 0.1,
            # num_train_epochs = 3,
            max_steps = args.max_steps,
            learning_rate = 5e-6,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.0,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = f"{args.output_dir}/{train_run_name}",
            save_steps = args.save_steps,
            report_to=None,
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=10,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,        
        ),
        beta = args.beta,
        train_dataset = raw_datasets["train"],
        eval_dataset = raw_datasets["valid"],
        tokenizer = tokenizer,
        max_length = max_seq_length,
        max_prompt_length = prompt_length,
    )


    dpo_trainer.train()

else:
    train_run_name = args.train_run_name


def extract_response(predict):
    # Split the string at "Response:\n" and return the part after it
    response = predict.split("Response:\n", 1)[1]
    return response

def evaluate_peft_model(model, sample, tokenizer):
    inputs = tokenizer(
    [   
        alpaca_prompt.format(
        sample["instruction"], # instruction
        sample["input"], # input
        "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    prediction = model.generate(**inputs, max_new_tokens = args.max_new_tokens, use_cache = True, pad_token_id=tokenizer.eos_token_id)
    prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    prediction = extract_response(prediction[0])
    print(sample["input"][0])
    print(sample["output"][0])
    return sample["input"], prediction, sample["output"]

datasets = load_dataset(args.dataset_dir, data_files={"test": args.test_dataset_name})

for save_step in range(args.save_steps, args.max_steps + 1, args.save_steps):
    if True:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "{}/{}/checkpoint-{}".format(args.output_dir, train_run_name, save_step),
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    
    FastLanguageModel.for_inference(model)

    path_name = os.path.join(f"./{args.predict_dir}",train_run_name)

    if not os.path.exists(path_name):
        os.makedirs(path_name)
        print(f"Create path: {path_name}")
    else:
        print(f"Path exists: {path_name}")

    for data_split in ['test']:
        output = []
        for sample in datasets[data_split]:
            input, p,l = evaluate_peft_model(model, sample, tokenizer)
            output.append({
            "input": input,
            "output": l,
            "predict": p,
                })
            # break

        with open(os.path.join(path_name, 'meta_llama3_8B_{}_{}.jsonl'.format(data_split, save_step)), 'w') as f:
            json.dump(output, f, indent=4)
        
        print("{}-complete!".format(save_step))



