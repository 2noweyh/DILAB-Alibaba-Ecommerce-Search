#!/bin/bash

set -e

# ======================
# Set route
# ======================
DATA_DIR="./data/raw"
REFINE_DIR="./data/refine"
OUT_DIR="./data/preprocessed"
mkdir -p $OUT_DIR
DEVICE=0

# ======================
# QC preprocessing pipeline
# ======================
run_QC_preprocess() {
    echo "[STEP 1] QC Tagging"
    python ./src/data_preprocess/qc_tagging.py \
        --input $REFINE_DIR/train_QC_refined.jsonl \
        --output $OUT_DIR/train_QC_tagged.jsonl

    python ./src/data_preprocess/qc_tagging.py \
        --input $DATA_DIR/test_QC.txt \
        --output $OUT_DIR/test_QC_tagged.jsonl

    echo "[STEP 2] Convert QC format"
    python ./src/data_preprocess/qc_convert_format.py \
        --input $OUT_DIR/train_QC_tagged.jsonl \
        --split train \
        --output $OUT_DIR/train_QC_final.txt

    python ./src/data_preprocess/qc_convert_format.py \
        --input $OUT_DIR/test_QC_tagged.jsonl \
        --split test \
        --output $OUT_DIR/test_QC_final.txt
}

# ======================
# QI preprocessing pipeline
# ======================
run_QI_preprocess() {
    echo "[STEP 1] QI Tagging"
    python ./src/data_preprocess/qi_tagging.py \
        --input $REFINE_DIR/train_QI_refined.jsonl \
        --qc_path ./data/train_QC.txt \
        --output $OUT_DIR/train_QI_tagged.jsonl

    python ./src/data_preprocess/qi_tagging.py \
        --input $DATA_DIR/test_QI.txt \
        --qc_path ./data/train_QC.txt \
        --output $OUT_DIR/test_QI_tagged.jsonl

    echo "[STEP 2] vLLM-based caption generation"
    CUDA_VISIBLE_DEVICES=$DEVICE python ./src/data_preprocess/qi_caption_vllm.py \
        --input $OUT_DIR/train_QI_tagged.jsonl \
        --output $OUT_DIR/train_QI_captioned.jsonl

    CUDA_VISIBLE_DEVICES=$DEVICE python ./src/data_preprocess/qi_caption_vllm.py \
        --input $OUT_DIR/test_QI_tagged.jsonl \
        --output $OUT_DIR/test_QI_captioned.jsonl

    echo "[STEP 3] Convert QI format"
    python ./src/data_preprocess/qi_convert_format.py \
        --input $OUT_DIR/train_QI_captioned.jsonl \
        --split train \
        --output $OUT_DIR/train_QI_final.txt

    python ./src/data_preprocess/qi_convert_format.py \
        --input $OUT_DIR/test_QI_captioned.jsonl \
        --split test \
        --output $OUT_DIR/test_QI_final.txt
}

# ======================
# Run
# ======================
echo "===== Running QC preprocessing ====="
run_QC_preprocess
echo "===== Running QI preprocessing ====="
run_QI_preprocess
echo "===== All preprocessing completed ====="
