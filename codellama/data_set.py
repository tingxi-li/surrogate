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
            "is_reject": assistant_turn.get("is_reject", None)
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
            "rejected": assistant_turn.get("content", ""),
            "is_reject": assistant_turn.get("is_reject", None)
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
    
    if data_format == "sft":
        
        def func(examples):
            
            full_texts = []
            
            for prompt, response in zip(examples["prompt"], examples["response"]):

                full_text = f"{prompt}{response}{tokenizer.eos_token}"
                full_texts.append(full_text)
             
            model_inputs = tokenizer(
                full_texts,
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
            
            labels = []
            
            for i, (prompt, full_text) in enumerate(zip(examples["prompt"], full_texts)):

                prompt_ids = tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"][0]
                
                label = model_inputs["input_ids"][i].clone()
                label[:len(prompt_ids)] = -100
                labels.append(label)
            
            model_inputs["labels"] = torch.stack(labels)
            
            return model_inputs
        
    elif data_format == "dpo":

        def func(examples):
            # return examples
            chosen_texts = []
            rejected_texts = []
            
            for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
                chosen_text = f"{prompt}{chosen}{tokenizer.eos_token}"
                rejected_text = f"{prompt}{rejected}{tokenizer.eos_token}"
                
                chosen_texts.append(chosen_text)
                rejected_texts.append(rejected_text)
            
            chosen_inputs = tokenizer(
                chosen_texts,
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
            
            rejected_inputs = tokenizer(
                rejected_texts,
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
            
            
            labels = []
            
            for i, prompt in enumerate(examples["prompt"]):
                prompt_ids = tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"][0]
                
                chosen_label = chosen_inputs["input_ids"][i].clone()
                chosen_label[:len(prompt_ids)] = -100
                rejected_label = rejected_inputs["input_ids"][i].clone()
                rejected_label[:len(prompt_ids)] = -100
                
                labels.append((chosen_label, rejected_label))
                
            chosen_labels, rejected_labels = zip(*labels)
            chosen_labels = torch.stack(chosen_labels)
            rejected_labels = torch.stack(rejected_labels)
            model_inputs = {
                "chosen_input_ids": chosen_inputs["input_ids"],
                "chosen_attention_mask": chosen_inputs["attention_mask"],
                "rejected_input_ids": rejected_inputs["input_ids"],
                "rejected_attention_mask": rejected_inputs["attention_mask"],
                "chosen_labels": chosen_labels,
                "rejected_labels": rejected_labels
            }
            return model_inputs
            for i, (prompt, full_text) in enumerate(zip(examples["prompt"], full_texts)):

                prompt_ids = tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"][0]
                
                label = model_inputs["input_ids"][i].clone()
                label[:len(prompt_ids)] = -100
                labels.append(label)
            
            model_inputs["labels"] = torch.stack(labels)
            
            return model_inputs
    
    return func
    
    
def get_data_set(TRAIN_SIZE, VAL_SIZE, TEST_SIZE, base_dir=None, balanced=True, data_format = "sft", if_print=False):

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
                    
                    if data_format == "sft":
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


class DataCollatorForSFT:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        input_ids = []
        labels = []
        attention_mask = []
        
        for f in features:
            input_id = f["input_ids"] if isinstance(f["input_ids"], torch.Tensor) else torch.tensor(f["input_ids"])
            label = f["labels"] if isinstance(f["labels"], torch.Tensor) else torch.tensor(f["labels"])
            mask = f["attention_mask"] if isinstance(f["attention_mask"], torch.Tensor) else torch.tensor(f["attention_mask"])
            
            input_ids.append(input_id)
            labels.append(label)
            attention_mask.append(mask)
        
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
        
class DataCollatorForDPO:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        chosen_input_ids = []
        chosen_attention_mask = []
        chosen_labels = []
        rejected_input_ids = []
        rejected_attention_mask = []
        rejected_labels = []
        prompt_input_ids = []
        prompt_attention_mask = []

        for f in features:
            # Ensure inputs are tensors
            c_input_id = torch.tensor(f.get("chosen_input_ids", []), dtype=torch.long) if not isinstance(f.get("chosen_input_ids"), torch.Tensor) else f["chosen_input_ids"]
            c_attn_mask = torch.tensor(f.get("chosen_attention_mask", []), dtype=torch.long) if not isinstance(f.get("chosen_attention_mask"), torch.Tensor) else f["chosen_attention_mask"]
            c_label = torch.tensor(f.get("chosen_labels", []), dtype=torch.long) if not isinstance(f.get("chosen_labels"), torch.Tensor) else f["chosen_labels"]
            
            r_input_id = torch.tensor(f.get("rejected_input_ids", []), dtype=torch.long) if not isinstance(f.get("rejected_input_ids"), torch.Tensor) else f["rejected_input_ids"]
            r_attn_mask = torch.tensor(f.get("rejected_attention_mask", []), dtype=torch.long) if not isinstance(f.get("rejected_attention_mask"), torch.Tensor) else f["rejected_attention_mask"]
            r_label = torch.tensor(f.get("rejected_labels", []), dtype=torch.long) if not isinstance(f.get("rejected_labels"), torch.Tensor) else f["rejected_labels"]
            
            # Extract prompt_input_ids and prompt_attention_mask
            prompt_id = torch.tensor(f.get("prompt_input_ids", []), dtype=torch.long) if not isinstance(f.get("prompt_input_ids"), torch.Tensor) else f["prompt_input_ids"]
            prompt_mask = torch.tensor(f.get("prompt_attention_mask", []), dtype=torch.long) if not isinstance(f.get("prompt_attention_mask"), torch.Tensor) else f["prompt_attention_mask"]

            # Fix: Ensure attention mask has at least one valid token if non-empty
            if c_attn_mask.numel() == 0:
                c_attn_mask = torch.tensor([1], dtype=torch.long)
            elif c_attn_mask.sum() == 0:
                c_attn_mask[0] = 1

            if r_attn_mask.numel() == 0:
                r_attn_mask = torch.tensor([1], dtype=torch.long)
            elif r_attn_mask.sum() == 0:
                r_attn_mask[0] = 1

            if prompt_mask.numel() == 0:
                prompt_mask = torch.tensor([1], dtype=torch.long)
            elif prompt_mask.sum() == 0:
                prompt_mask[0] = 1
            
            chosen_input_ids.append(c_input_id)
            chosen_attention_mask.append(c_attn_mask)
            chosen_labels.append(c_label)
            rejected_input_ids.append(r_input_id)
            rejected_attention_mask.append(r_attn_mask)
            rejected_labels.append(r_label)
            prompt_input_ids.append(prompt_id)
            prompt_attention_mask.append(prompt_mask)

        # Pad sequences
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        chosen_attention_mask = torch.nn.utils.rnn.pad_sequence(
            chosen_attention_mask, batch_first=True, padding_value=0
        )
        chosen_labels = torch.nn.utils.rnn.pad_sequence(
            chosen_labels, batch_first=True, padding_value=-100
        )
        
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        rejected_attention_mask = torch.nn.utils.rnn.pad_sequence(
            rejected_attention_mask, batch_first=True, padding_value=0
        )
        rejected_labels = torch.nn.utils.rnn.pad_sequence(
            rejected_labels, batch_first=True, padding_value=-100
        )
        
        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        prompt_attention_mask = torch.nn.utils.rnn.pad_sequence(
            prompt_attention_mask, batch_first=True, padding_value=0
        )

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }
        
        
if __name__ == "__main__":
    def convert_json_to_jsonl(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)  
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)  
                outfile.write(json_line + '\n')

    get_data_set(0.9, 0.05, 0.05,
        base_dir="/home/ubuntu/surrogate/data",
        balanced=True,
        data_format="sft"
    )
    data_format = "sft"  # or "dpo", "rlhf"
    positive_data = []
    negative_data = []
    path = "/home/ubuntu/surrogate/data/_dev_set_modified_filtered.jsonl"
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                
                if data_format == "sft":
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
