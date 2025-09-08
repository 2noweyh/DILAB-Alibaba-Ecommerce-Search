#!/bin/bash
# bash inference.sh Qwen2.5-14B QC
# bash inference.sh Qwen2.5-14B QI

model=$1
task=$2

model_path="./models/${model}-finetuned-${task}"
data_path="./outputs/preprocessed/dev_${task}_final.txt"
output_path="./outputs/${task}_${model}_result.txt"
# output_path="./outputs/submit_${task}.txt"


# Base model & template 지정

if [ "$model" = "Qwen2.5-14B" ]; then
    base_model="Qwen/Qwen2.5-14B-Instruct"
    prompt_template_name="alpaca"
elif [ "$model" = "Llama-2-13B-chat" ]; then
    base_model="meta-llama/Llama-2-13b-chat-hf"
    prompt_template_name="alpaca"
elif [ "$model" = "Llama-2-7B-chat" ]; then
    base_model="meta-llama/Llama-2-7b-chat-hf"
    prompt_template_name="alpaca"
elif [ "$model" = "Mistral-7B-Instruct-v0.2" ]; then
    base_model="mistralai/Mistral-7B-Instruct-v0.2"
    prompt_template_name="mistral"
elif [ "$model" = "eCeLLM-S" ]; then
    base_model="NingLab/eCeLLM-S"
    prompt_template_name="alpaca"
elif [ "$model" = "eCeLLM-M" ]; then
    base_model="NingLab/eCeLLM-M"
    prompt_template_name="alpaca"
elif [ "$model" = "eCeLLM-L" ]; then
    base_model="NingLab/eCeLLM-L"
    prompt_template_name="alpaca"
else
    base_model=""
    prompt_template_name=""
fi

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model $base_model \
    --lora_weights $model_path \
    --data_path $data_path \
    --output_data_path $output_path \
    --prompt_template $prompt_template_name \
    --save_scores False
