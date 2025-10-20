#!/bin/bash
# bash script/train.sh 2 Qwen2.5-14B QC
# bash script/train.sh 2 Qwen2.5-14B QI

num_epochs=$1
model=$2
task=$3

output_dir="./model/${model}-finetuned-${task}"
data_path="./data/preprocessed/train_${task}_final.txt"
mkdir -p $output_dir
DEVICE=0

if [ "$model" = "Qwen2.5-14B" ]; then
    base_model="Qwen/Qwen2.5-14B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca"
elif [ "$model" = "Meta-Llama-3-8B" ]; then 
    base_model="meta-llama/Meta-Llama-3-8B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca" 
elif [ "$model" = "Llama-2-13B" ]; then
    base_model="meta-llama/Llama-2-13b-chat-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca"
elif [ "$model" = "Llama-2-7B" ]; then
    base_model="meta-llama/Llama-2-7b-chat-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca"
elif [ "$model" = "Mistral-7B" ]; then
    base_model="mistralai/Mistral-7B-Instruct-v0.2"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="mistral"
elif [ "$model" = "eCeLLM-S" ]; then
    base_model="NingLab/eCeLLM-S"
    lora_target_modules='[Wqkv, out_proj, fc1, fc2, linear]'
    prompt_template_name="alpaca"
elif [ "$model" = "eCeLLM-M" ]; then
    base_model="NingLab/eCeLLM-M"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca"
elif [ "$model" = "eCeLLM-L" ]; then
    base_model="NingLab/eCeLLM-L"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca"
else
    base_model=""
    lora_target_modules=""
    prompt_template_name=""
fi

# export CUDA_LAUNCH_BLOCKING = 1  # Uncomment if it fails to run (serial execution) 
CUDA_VISIBLE_DEVICES=$DEVICE python ./src/train.py \
    --base_model $base_model \
    --data_path $data_path \
    --output_dir $output_dir \
    --batch_size 8 \
    --micro_batch_size 2 \
    --num_epochs $num_epochs \
    --cutoff_len 1024 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "$lora_target_modules" \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name $prompt_template_name \
    --lr_scheduler 'cosine' \
    --optim "adamw_torch" \
    --warmup_ratio 0.05