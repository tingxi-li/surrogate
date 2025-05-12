import os
import json
import torch
import random
import logging
from datasets import Dataset
from constants import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RM_COLS
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling


def extract_data(entry):
    pos = []
    neg = []
    turns = entry.get("turns", [])
    for turn_id in range(0, len(turns), 2):
        if turn_id + 1 >= len(turns):
            break
        
        user_turn = turns[turn_id]
        assistant_turn = turns[turn_id + 1]
        datapoint = {
            "prompt": user_turn.get("content", ""),
            "response": assistant_turn.get("content", ""),
        }
        
        is_positive = assistant_turn.get("is_reject", None)
        
        if is_positive == True:
            pos.append(datapoint)
        elif is_positive == False:
            neg.append(datapoint)
        else:
            continue

    return pos, neg


def extract_dpo_data(entry):
    pos = []
    neg = []
    turns = entry.get("turns", [])
    for turn_id in range(0, len(turns), 2):
        if turn_id + 1 >= len(turns):
            break
        
        user_turn = turns[turn_id]
        assistant_turn = turns[turn_id + 1]
        datapoint = {
            "prompt": user_turn.get("content", ""),
            "chosen": assistant_turn.get("dpo", ""),
            "rejected": assistant_turn.get("content", "") 
        }
        
        is_positive = assistant_turn.get("is_reject", None)
        
        if is_positive == True:
            pos.append(datapoint)
        elif is_positive == False:
            neg.append(datapoint)
        else:
            continue

    return pos, neg


def extract_rlhf_data(entry):
    pass


def get_tokenize_fn(tokenizer, data_format):
    if data_format == "completion":
        def func(examples):
            # 
            full_texts = []
            for prompt, completion in zip(examples["prompt"], examples["response"]):
                # 
                full_text = f"{prompt}{completion}{tokenizer.eos_token}"
                full_texts.append(full_text)
            
            # 
            model_inputs = tokenizer(
                full_texts,
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
            
            #
            labels = []
            for i, (prompt, full_text) in enumerate(zip(examples["prompt"], full_texts)):
                # 
                prompt_ids = tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"][0]
                
                # 
                label = model_inputs["input_ids"][i].clone()
                label[:len(prompt_ids)] = -100
                labels.append(label)
            
            model_inputs["labels"] = torch.stack(labels)
            
            return model_inputs
    
    return func
    
    
def get_data_set(TRAIN_SIZE, VAL_SIZE, TEST_SIZE, base_dir=None, balanced=True, data_format = "completion", if_print=False):

    paths = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".jsonl"):
                file_path = os.path.join(root, f)
                paths.append(file_path)
    print(f"Found {len(paths)} JSONL files in {base_dir}") if if_print else None

    for path in paths:
        print(" "*4,f" - {path}") if if_print else None

    positive_data = []
    negative_data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    if data_format == "completion":
                        pos, neg = extract_data(entry)
                    elif data_format == "dpo":
                        pos, neg = extract_dpo_data(entry)
                    elif data_format == "rlhf":
                        pos, neg = extract_rlhf_data(entry)
                    else:
                        raise ValueError(f"Unknown data format: {data_format}")
                        
                    positive_data.extend(pos)
                    negative_data.extend(neg)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}") if if_print else None
    
    print(f"Extracted {len(positive_data)} positive and {len(negative_data)} negative samples. Total: {len(positive_data) + len(negative_data)}") if if_print else None
    
    total_data = positive_data + negative_data
    
    positive_count = len(positive_data)
    negative_count = len(negative_data)
    
    balanced_data = []
    if positive_count > negative_count:
        balanced_data.extend(negative_data)
        balanced_data.extend(random.sample(positive_data, negative_count))
    else:
        balanced_data.extend(positive_data)
        balanced_data.extend(random.sample(negative_data, positive_count))
       
    random.shuffle(total_data) 
    random.shuffle(balanced_data)
    
    if balanced:
        print(f"return balanced data") if if_print else None
        train_len = int(len(balanced_data) * TRAIN_SIZE)
        val_len = int(len(balanced_data) * VAL_SIZE)
        train_set = Dataset.from_list(balanced_data[:train_len])
        eval_set = Dataset.from_list(balanced_data[train_len:train_len + val_len])
        test_set = Dataset.from_list(balanced_data[train_len + val_len:])
    else:
        print(f"return unbalanced data") if if_print else None
        train_len = int(len(total_data) * TRAIN_SIZE)
        val_len = int(len(total_data) * VAL_SIZE)
        train_set = Dataset.from_list(total_data[:train_len])
        eval_set = Dataset.from_list(total_data[train_len:train_len + val_len])
        test_set = Dataset.from_list(total_data[train_len + val_len:])
    
    return train_set, eval_set, test_set


class DataCollatorForPromptCompletion:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # 获取输入和标签，确保是 Tensor 类型
        input_ids = []
        labels = []
        attention_mask = []
        
        for f in features:
            # 确保转换为 tensor
            input_id = f["input_ids"] if isinstance(f["input_ids"], torch.Tensor) else torch.tensor(f["input_ids"])
            label = f["labels"] if isinstance(f["labels"], torch.Tensor) else torch.tensor(f["labels"])
            mask = f["attention_mask"] if isinstance(f["attention_mask"], torch.Tensor) else torch.tensor(f["attention_mask"])
            
            input_ids.append(input_id)
            labels.append(label)
            attention_mask.append(mask)
        
        # 填充序列
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        
if __name__ == "__main__":
    def convert_json_to_jsonl(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)  
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)  
                outfile.write(json_line + '\n')

    load_dataset(0.9, 0.05, 0.05,
        base_dir="/home/ubuntu/surrogate/data",
        balanced=True,
        data_format="completion"
    )
    data_format = "completion"  # or "dpo", "rlhf"
    positive_data = []
    negative_data = []
    path = "/home/ubuntu/surrogate/data/_dev_set_modified_filtered.jsonl"
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                
                if data_format == "completion":
                    pos, neg = extract_data(entry)
                elif data_format == "dpo":
                    pos, neg = extract_dpo_data(entry)
                elif data_format == "rlhf":
                    pos, neg = extract_rlhf_data(entry)
                else:
                    raise ValueError(f"Unknown data format: {data_format}")
                    
                positive_data.extend(pos)
                negative_data.extend(neg)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        
        print(f"Extracted {len(positive_data)} positive and {len(negative_data)} negative samples. Total: {len(positive_data) + len(negative_data)}")
