# # 7b SFT 
# torchrun --standalone --nproc_per_node=8 train.py \
#     --model_name codellama/CodeLlama-7b-hf \
#     --balanced \
#     --data_format sft || true


# # 7b DPO 
# torchrun --standalone --nproc_per_node=8 train.py \
#     --model_name codellama/CodeLlama-7b-hf \
#     --balanced \
#     --data_format dpo || true


# # 7b DPO Python
# torchrun --standalone --nproc_per_node=8 train.py \
#     --model_name codellama/CodeLlama-7b-Python-hf \
#     --balanced \
#     --data_format dpo || true


# # 7b SFT Python
# torchrun --standalone --nproc_per_node=8 train.py \
#     --model_name codellama/CodeLlama-7b-Python-hf \
#     --balanced \
#     --data_format sft || true


# 7b SFT Instruct
torchrun --standalone --nproc_per_node=8 train.py \
    --model_name codellama/CodeLlama-7b-Instruct-hf \
    --balanced \
    --data_format sft || true


# 7b DPO Instruct 
torchrun --standalone --nproc_per_node=8 train.py \
    --model_name codellama/CodeLlama-7b-Instruct-hf \
    --balanced \
    --data_format dpo || true