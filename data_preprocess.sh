#!/bin/bash
# bash data_preprocess.bash

set -e  # 에러 발생 시 중단

# ======================
# 경로 설정
# ======================
DATA_DIR="./data"
OUT_DIR="./outputs/preprocessed"
mkdir -p $OUT_DIR

# ======================
# QC 전처리 파이프라인
# ======================
run_QC_preprocess() {
    echo "[STEP 1] QC 태깅"
    python ./data_preprocess/qc_tagging.py \
        --input $DATA_DIR/train_QC.txt \
        --output $OUT_DIR/train_QC_tagged.jsonl

    python ./data_preprocess/qc_tagging.py \
        --input $$DATA_DIR/dev_QC.txt \
        --output $OUT_DIR/dev_QC_tagged.jsonl

    echo "[STEP 2] QC 포맷 변환"
    python ./data_preprocess/qc_convert_format.py \
        --input $OUT_DIR/train_QC_tagged.jsonl \
        --split train \
        --output $OUT_DIR/train_QC_final.txt

    python ./data_preprocess/qc_convert_format.py \
        --input $OUT_DIR/dev_QC_tagged.jsonl \
        --split dev \
        --output $OUT_DIR/dev_QC_final.txt
}

# ======================
# QI 전처리 파이프라인
# ======================
run_QI_preprocess() {
    echo "[STEP 1] QI 태깅"
    python ./data_preprocess/qi_tagging.py \
        --input $DATA_DIR/train_QI.txt \
        --qc_path ./data/train_QC.txt \
        --output $OUT_DIR/train_QI_tagged.jsonl

    python ./data_preprocess/qi_tagging.py \
        --input $DATA_DIR/dev_QI.txt \
        --qc_path ./data/train_QC.txt \
        --output $OUT_DIR/dev_QI_tagged.jsonl

    echo "[STEP 2] vLLM 기반 캡션 생성"
    CUDA_VISIBLE_DEVICES=1 python ./data_preprocess/qi_caption_vllm.py \
        --input $OUT_DIR/train_QI_tagged.jsonl \
        --output $OUT_DIR/train_QI_captioned.jsonl

    CUDA_VISIBLE_DEVICES=1 python ./data_preprocess/qi_caption_vllm.py \
        --input $OUT_DIR/dev_QI_tagged.jsonl \
        --output $OUT_DIR/dev_QI_captioned.jsonl

    echo "[STEP 3] QI 포맷 변환"
    python ./data_preprocess/qi_convert_format.py \
        --input $OUT_DIR/train_QI_captioned.jsonl \
        --split train \
        --output $OUT_DIR/train_QI_final.jsonl

    python ./data_preprocess/qi_convert_format.py \
        --input $OUT_DIR/dev_QI_captioned.jsonl \
        --split dev \
        --output $OUT_DIR/dev_QI_final.jsonl
}

# ======================
# 실행
# ======================
echo "===== QC 전처리 실행 ====="
run_QC_preprocess
echo "===== QI 전처리 실행 ====="
run_QI_preprocess
echo "===== 모든 전처리 완료 ====="
