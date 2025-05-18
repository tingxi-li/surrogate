# # 13b SFT 
# torchrun --standalone --nproc_per_node=8 train.py \
#     --model_name codellama/CodeLlama-13b-hf \
#     --balanced \
#     --data_format sft || true


# # 13b DPO 
# torchrun --standalone --nproc_per_node=8 train.py \
#     --model_name codellama/CodeLlama-13b-hf \
#     --balanced \
#     --data_format dpo || true


# 13b DPO Python
torchrun --standalone --nproc_per_node=8 train.py \
    --model_name codellama/CodeLlama-13b-Python-hf \
    --balanced \
    --output_dir /opt/dlami/nvme/surrogate_ckpt/13b \
    --data_format dpo || true


# 13b SFT Python
torchrun --standalone --nproc_per_node=8 train.py \
    --model_name codellama/CodeLlama-13b-Python-hf \
    --balanced \
    --output_dir /opt/dlami/nvme/surrogate_ckpt/13b \
    --data_format sft || true


# # 13b SFT Instruct
# torchrun --standalone --nproc_per_node=8 train.py \
#     --model_name codellama/CodeLlama-13b-Instruct-hf \
#     --balanced \
#     --data_format sft || true


# # 13b DPO Instruct 
# torchrun --standalone --nproc_per_node=8 train.py \
#     --model_name codellama/CodeLlama-13b-Instruct-hf \
#     --balanced \
#     --data_format dpo || true