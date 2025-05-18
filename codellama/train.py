import os
import pdb
import json
import torch
import random
import argparse
import deepspeed
import transformers
from constants import *
from transformers import (
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datetime import datetime
import torch.distributed as dist
from trl import DPOConfig, DPOTrainer
from torch.utils.tensorboard import SummaryWriter
from data_set import get_data_set, get_tokenize_fn, DataCollatorForSFT, DataCollatorForDPO

random.seed(0)
torch.manual_seed(0)
iam_rank0 = os.getenv("RANK", "0") == "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

local_rank = int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", init_method="env://")

parser = argparse.ArgumentParser(description="Train a model with LoRA.")

parser.add_argument("--base_dir", type=str, default="/home/ubuntu/surrogate/data")
parser.add_argument('--balanced', action='store_true', help="Use balanced dataset.")
parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-hf")
parser.add_argument("--output_dir", type=str, default="/opt/dlami/nvme/surrogate_ckpt/7b")
parser.add_argument("--data_format", type=str, default="sft")

args = parser.parse_args()

model_name = args.model_name
output_dir = args.output_dir
data_format = args.data_format
base_dir = args.base_dir
balanced = args.balanced

if iam_rank0:
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(output_dir, f"{time_stamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
train_set, eval_set, test_set = get_data_set(TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
                                            base_dir=base_dir,
                                            balanced=balanced,
                                            data_format=data_format,
                                            if_print=iam_rank0)

tokenize_fn = get_tokenize_fn(tokenizer, data_format)

tokenized_train_set = train_set.map(
    tokenize_fn,
    batched=True,
    remove_columns=train_set.column_names,
    desc="Tokenizing train dataset"
)

tokenized_eval_set = eval_set.map(
    tokenize_fn,
    batched=True,
    remove_columns=eval_set.column_names,
    desc="Tokenizing eval dataset"
)

tokenized_test_set = test_set.map(
    tokenize_fn,
    batched=True,
    remove_columns=test_set.column_names,
    desc="Tokenizing test dataset"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  
    use_cache=False,  
)

if data_format == "sft":
    data_collator = DataCollatorForSFT(tokenizer)
elif data_format == "dpo":
    data_collator = DataCollatorForDPO(tokenizer)
elif data_format == "rlhf":
    pass
else:
    raise ValueError(f"Unsupported data format: {data_format}")




if data_format == "sft":
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,  
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  
        gradient_checkpointing=True,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        eval_strategy="steps" if eval_set else "no",
        eval_steps=100 if eval_set else None,
        logging_dir=os.path.join(output_dir, "logs"),
        warmup_steps=100,
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        load_best_model_at_end=True if eval_set else False,
        metric_for_best_model="eval_loss" if eval_set else None,
        greater_is_better=False,
        report_to="tensorboard",
        deepspeed="./ds_config.json",  
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_eval_set,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
elif data_format == "dpo":
    
    training_args = DPOConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,  
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  
        gradient_checkpointing=True,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        eval_strategy="steps" if eval_set else "no",
        eval_steps=100 if eval_set else None,
        logging_dir=os.path.join(output_dir, "logs"),
        warmup_steps=100,
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        load_best_model_at_end=True if eval_set else False,
        metric_for_best_model="eval_loss" if eval_set else None,
        greater_is_better=False,
        report_to="tensorboard",
        deepspeed="./ds_config.json",  
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    
    trainer = DPOTrainer(
        model=model, 
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
    )

elif data_format == "rlhf":
    # RLHF training logic goes here
    pass
else:
    raise ValueError(f"Unsupported data format: {data_format}")

if iam_rank0:
    config = {
        "args": vars(args),
        "training_arguments": training_args.to_dict(),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to {config_path}")
    
if __name__ == "__main__":
    trainer.train()
