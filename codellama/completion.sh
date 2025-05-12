#!/bin/bash

torchrun --standalone --nproc_per_node=8 train.py \
    --model_name codellama/CodeLlama-7b-hf \
    --balanced True \
    --data_format completion || true

torchrun --standalone --nproc_per_node=8 train.py \
    --model_name codellama/CodeLlama-13b-hf \
    --balanced True \
    --data_format completion || true

torchrun --standalone --nproc_per_node=8 train.py \
    --model_name codellama/CodeLlama-7b-hf \
    --balanced False \
    --data_format completion || true

torchrun --standalone --nproc_per_node=8 train.py \
    --model_name codellama/CodeLlama-13b-hf \
    --balanced False \
    --data_format completion || true
    