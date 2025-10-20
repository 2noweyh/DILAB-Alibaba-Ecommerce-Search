#!/bin/bash
# bash quality_refinement.sh

set -e

# ======================
# Set route
# ======================
num_epochs=2
DEVICE=0 
model="eCeLLM-M"

DATA_DIR="./data/raw"
OUT_DIR="./data/refine"
base_model="NingLab/eCeLLM-M"
lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
prompt_template_name="alpaca"

# ======================
# Start
# ======================
for task in QC QI; do
    echo "==============================="
    echo ">>> Start running: $task (epochs=$num_epochs, device=$DEVICE)"
    echo "==============================="

    # ======================
    # 1. Data Cleaning (ICL Conversion)
    # ======================
    echo ">>> [STEP 1-1] Start data formating ($task)"
    mkdir -p $OUT_DIR
    DATA_PATH="$OUT_DIR/train_${task}_ecellm.txt"
    if [ "$task" = "QC" ]; then
        python ./src/quality_refinement/quality_format.py \
            --run convert_qc_format_with_icl \
            --input $DATA_DIR/train_QC.txt \
            --output $DATA_PATH
    elif [ "$task" = "QI" ]; then
        python ./src/quality_refinement/quality_format.py \
            --run convert_qi_format_with_icl \
            --input $DATA_DIR/train_QI.txt \
            --output $DATA_PATH
    else
        echo "Task must be QI or QC"
        exit 1
    fi

    # # ======================
    # # 2. Train 
    # # ======================
    echo ">>> [STEP 1-2] Start Training ($model-$task)"
    MODEL_PATH="./model/${model}-finetuned-${task}"
    mkdir -p $MODEL_PATH

    # export CUDA_LAUNCH_BLOCKING=1  # Uncomment if it fails to run (serial execution) 
    CUDA_VISIBLE_DEVICES=$DEVICE python ./src/train.py \
        --base_model $base_model \
        --data_path $DATA_PATH \
        --output_dir $MODEL_PATH \
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

    # ======================
    # 3. Inference
    # ======================
    echo ">>> [STEP 1-3] Start inference ($task)"
    OUTPUT_PATH="${OUT_DIR}/train_${task}_ecellm_result.txt"
    
    CUDA_VISIBLE_DEVICES=$DEVICE python ./src/predict.py \
        --base_model $base_model \
        --lora_weights $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_data_path $OUTPUT_PATH \
        --prompt_template $prompt_template_name \
        --save_scores True

    # ======================
    # 4. Refinement
    # ======================
    echo "[STEP 2] Start Refinement ($task)"

    if [ "$task" = "QC" ]; then
        python ./src/quality_refinement/quality_cleaning.py \
            --task $task \
            --input $DATA_DIR/train_QC.txt \
            --output $OUT_DIR/train_QC_refined.jsonl\
            --pred $OUTPUT_PATH
    elif [ "$task" = "QI" ]; then
        python ./src/quality_refinement/quality_cleaning.py \
            --task $task \
            --input $DATA_DIR/train_QI.txt \
            --output $OUT_DIR/train_QI_refined.jsonl \
            --pred $OUTPUT_PATH
    else
        echo "Task must be QI or QC"
        exit 1
    fi

    echo "✅ Done created"
    echo ">>> GPU Waiting for memory cleanup"
    sleep 5

done