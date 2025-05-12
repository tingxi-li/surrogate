import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pdb

# Verify the HF_HOME setting
print(f"HF_HOME is set to: {os.environ.get('HF_HOME', 'Not set')}")

def load_model(model_id=None):
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use half precision for efficiency
        device_map="auto"  # Automatically manage device placement
    )
    
    return model, tokenizer

def generate_code(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and generate code with CodeLlama.")
    parser.add_argument("--model_id", type=str, default="codellama/CodeLlama-7b-hf", help="Model ID to load.")
    model_id = parser.parse_args().model_id
    model, tokenizer = load_model(model_id)
    # _, _ = load_model("codellama/CodeLlama-7b-Python-hf")
    # _, _ = load_model("codellama/CodeLlama-7b-Instruct-hf")