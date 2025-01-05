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
from datasets import load_dataset
import wandb, os


parser = argparse.ArgumentParser(description="Set parameters.")
parser.add_argument('--dataset_dir', type=str, help='Path to the dataset directory.', default="/home/jtan/LLM-QO/data")
parser.add_argument('--train_dataset_name', type=str, help='Name of the training dataset.', default="train.jsonl")
parser.add_argument('--valid_dataset_name', type=str, help='Name of the validation dataset.', default="valid.jsonl")
parser.add_argument('--test_dataset_name', type=str, help='Name of the testing dataset.', default="test.jsonl")
parser.add_argument('--max_steps', type=int, help='Max number of training steps.', default=1500)
parser.add_argument('--save_steps', type=int, help='Number of save steps .', default=300)
parser.add_argument('--llm_name', type=str, help='Name of the language model.', default="llama3-8b")
parser.add_argument('--max_new_tokens', type=int, help='The maximum numbers of tokens to generate.', default=128)
parser.add_argument('--predict_dir', type=str, help='Name of the predict directory.', default="predicts_dsb")
parser.add_argument('--output_dir', type=str, help='Name of the output directory.', default="outputs_dsb")
parser.add_argument('--eval_mode', type=str, help='Whether only to eval.', default="false")
parser.add_argument('--ckpt_dir', type=str, help='The checkpint directory.', default="ckpt_dir")


args = parser.parse_args()

print(is_bfloat16_supported)

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

#Load base model: llama3-8b
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
    #unsloth/llama-2-7b-bnb-4bit
    #unsloth/codellama-7b-bnb-4bit
] # More models at https://huggingface.co/unsloth

if args.llm_name == "llama3-8b":
    llm_name = "unsloth/llama-3-8b-bnb-4bit"
elif args.llm_name == "llama2-7b":
    llm_name = "unsloth/llama-2-7b-bnb-4bit"
elif args.llm_name == "codellama-7b":
    llm_name = "unsloth/codellama-7b-bnb-4bit"
elif args.llm_name == "mistral-7b":
    llm_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
else:
    print("Error model name!")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = llm_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


dataset = load_dataset(args.dataset_dir, data_files={"train": args.train_dataset_name, "valid": args.valid_dataset_name})

dataset = dataset.map(formatting_prompts_func, batched = True,)

#shuffle the dataset
shuffled_train_dataset = dataset["train"].shuffle(seed=42)  # You can set a seed for reproducibility

dataset["train"] = shuffled_train_dataset

train_dataset = dataset["train"]
eval_dataset = dataset["valid"]

if args.eval_mode == "false":

    data_dir = args.dataset_dir
    data_dir = data_dir.split("/")[1]
    train_data_name = args.train_dataset_name.split(".jsonl")[0]

    train_run_name = "SFT_" + data_dir + "_" + train_data_name + "_"+ args.llm_name + "-"

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["valid"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = args.max_steps,
            # num_train_epochs =3000,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            data_seed= 3407,
            output_dir = f"{args.output_dir}/{train_run_name}",
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=50,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            # Save checkpoint every X updates steps
            save_steps = args.save_steps,
        ),
    )

    trainer_stats = trainer.train()

else:
    train_run_name = args.train_run_name

def extract_response(predict):
    # Split the string at "Response:\n" and return the part after it
    response = predict.split("Response:\n", 1)[1]
    return response

def evaluate_peft_model(model, sample,top_p=0,top_k=0,tmp=0):
    inputs = tokenizer(
    [   
        alpaca_prompt.format(
        sample["instruction"], # instruction
        sample["input"], # input
        "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    if top_p==0 and top_k == 0:
        prediction = model.generate(**inputs, max_new_tokens = args.max_new_tokens, use_cache = True, pad_token_id=tokenizer.eos_token_id)
    else:
        prediction = model.generate(**inputs, max_new_tokens = args.max_new_tokens, use_cache = True, pad_token_id=tokenizer.eos_token_id, top_k=top_k, top_p = top_p,temperature=tmp)

    prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    prediction = extract_response(prediction[0])

    return sample["input"], prediction, sample["output"]

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

    dataset = load_dataset(args.dataset_dir, data_files={"test":args.test_dataset_name})

    path_name = os.path.join(f"./{args.predict_dir}",train_run_name)

    if not os.path.exists(path_name):
        os.makedirs(path_name)
        print(f"Create path: {path_name}")
    else:
        print(f"Path exists: {path_name}")
        
    num = 0

    for data_split in ['test']:
        output = []
        for sample in dataset[data_split]:
                        
            # break
            input, p,l = evaluate_peft_model(model, sample)
            output.append({
            "input": input,
            "output": l,
            "predict": p,
                })

            num += 1

        with open(os.path.join(path_name, 'meta_llama3_8B_{}_{}.jsonl'.format(data_split, save_step)), 'w') as f:
            json.dump(output, f, indent=4)
        
        print("{}-complete!".format(save_step))