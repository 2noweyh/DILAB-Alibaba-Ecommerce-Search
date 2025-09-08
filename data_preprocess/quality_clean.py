#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quality cleaning script for QI / QC datasets.

- Detects label mismatches using TF-IDF cosine + Jaccard similarity
- Uses model predictions (optional) to reinforce detection
- Auto-tunes thresholds per language unless disabled
- Flags suspect samples and outputs refined JSONL/CSV
"""

import json, re, os, math, argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# =========================
# 키워드 가드
# =========================
FP_BIAS_TERMS = [
    r"\b1/64\b", r"\b1:64\b", r"\bfigure\b", r"\bdiecast\b", r"\bpgm\b", r"\brwb\b",
    r"\bconverter\b", r"\badapter\b", r"\b75mm\b", r"\b12v\b", r"\b220v\b", r"\btransformer\b"
]
FN_BIAS_TERMS = [
    r"\binch\b", r"\bmonitor\b", r"\bcapacitor\b", r"\b(u|µ)?f\b", r"\bm(ah|a)\b",
    r"\b4k\b", r"\b128gb\b", r"\bgame\s*stick\b", r"\bhdmi\b", r"\bvga\b"
]

HI_COS_DEFAULT, HI_JAC_DEFAULT = 0.35, 0.20
LO_COS_DEFAULT, LO_JAC_DEFAULT = 0.12, 0.05
P_HI, P_LO = 0.90, 0.10

# =========================
# 함수
# =========================
def normalize_text(s: str):
    s = s or ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9가-힣áéíóúñüàèìòùçäöüß\s/.\-:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def jaccard(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows

def safe_get_prob1(row: dict) -> Optional[float]:
    if "prob1" in row and row["prob1"] is not None:
        try:
            return float(row["prob1"])
        except Exception:
            pass
    if "logits" in row and isinstance(row["logits"], (list, tuple)) and len(row["logits"]) >= 2:
        l0, l1 = float(row["logits"][0]), float(row["logits"][1])
        m = max(l0, l1)
        e0, e1 = math.exp(l0 - m), math.exp(l1 - m)
        return e1 / (e0 + e1 + 1e-12)
    return None

def any_regex_match(patterns: List[str], text: str) -> bool:
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False

def build_thresholds(df, auto_tune=True):
    lang_thresholds: Dict[str, Dict[str, float]] = {}
    if auto_tune:
        for lang, g in df.groupby("lang"):
            g0 = g[g["label"]=="0"]
            g1 = g[g["label"]=="1"]

            if len(g0) >= 100:
                hi_cos = float(np.quantile(g0["tfidf_cos"].values, P_HI))
                hi_jac = float(np.quantile(g0["jac"].values, P_HI))
            else:
                hi_cos, hi_jac = HI_COS_DEFAULT, HI_JAC_DEFAULT

            if len(g1) >= 100:
                lo_cos = float(np.quantile(g1["tfidf_cos"].values, P_LO))
                lo_jac = float(np.quantile(g1["jac"].values, P_LO))
            else:
                lo_cos, lo_jac = LO_COS_DEFAULT, LO_JAC_DEFAULT

            hi_cos = max(hi_cos, HI_COS_DEFAULT * 0.6)
            hi_jac = max(hi_jac, HI_JAC_DEFAULT * 0.6)
            lo_cos = min(lo_cos, HI_COS_DEFAULT)
            lo_jac = min(lo_jac, HI_JAC_DEFAULT)

            lang_thresholds[lang] = {
                "HI_COS": hi_cos, "HI_JAC": hi_jac,
                "LO_COS": lo_cos, "LO_JAC": lo_jac
            }
    else:
        for lang in df["lang"].unique():
            lang_thresholds[lang] = {
                "HI_COS": HI_COS_DEFAULT, "HI_JAC": HI_JAC_DEFAULT,
                "LO_COS": LO_COS_DEFAULT, "LO_JAC": LO_JAC_DEFAULT
            }
    return lang_thresholds

def flag_suspect_row(row, thresholds, use_pred=True):
    thr = thresholds.get(row["lang"], {
        "HI_COS": HI_COS_DEFAULT, "HI_JAC": HI_JAC_DEFAULT,
        "LO_COS": LO_COS_DEFAULT, "LO_JAC": LO_JAC_DEFAULT
    })
    hi_cos, hi_jac = thr["HI_COS"], thr["HI_JAC"]
    lo_cos, lo_jac = thr["LO_COS"], thr["LO_JAC"]

    cs = float(row["tfidf_cos"]); jc = float(row["jac"])
    label = row["label"]

    if label == "0":
        base_flag = (cs >= hi_cos) or (jc >= hi_jac)
    else:
        base_flag = (cs < lo_cos) and (jc < lo_jac)

    if use_pred:
        if row.get("disagree", False):
            base_flag = True
        prob1 = row.get("prob1", None)
        if prob1 is not None:
            if label == "0" and prob1 >= 0.9:
                base_flag = True
            if label == "1" and prob1 <= 0.1:
                base_flag = True

    text_all = f'{row["q_norm"]} || {row["i_norm"]}'
    if label == "0":
        if any_regex_match(FP_BIAS_TERMS, text_all):
            if cs >= max(hi_cos * 0.9, hi_cos - 0.02) or jc >= max(hi_jac * 0.9, hi_jac - 0.02):
                base_flag = True
    else:
        if any_regex_match(FN_BIAS_TERMS, text_all):
            if cs < lo_cos * 1.1 and jc < lo_jac * 1.1:
                base_flag = True

    return bool(base_flag)

# =========================
# 메인
# =========================
def main(args):
    # 데이터 로드
    rows = load_jsonl(args.input)

    df_rows = []
    for r in rows:
        if args.task.upper() == "QI":
            q = r.get("origin_query", "") or r.get("query", "")
            i = r.get("item_title", "")
        elif args.task.upper() == "QC":
            q = r.get("origin_query", "") or r.get("query", "")
            i = r.get("category_path", "")
        else:
            raise ValueError("task must be QI or QC")

        df_rows.append({
            "id": str(r.get("id")),
            "lang": r.get("language", "unk"),
            "q": q,
            "i": i,
            "label": str(r.get("label")),
        })

    df = pd.DataFrame(df_rows)
    df = df.dropna(subset=["q", "i", "label"])
    df = df[df["label"].isin(["0", "1"])].copy()

    # 정규화
    df["q_norm"] = df["q"].map(normalize_text)
    df["i_norm"] = df["i"].map(normalize_text)
    df["q_tok"] = df["q_norm"].str.split()
    df["i_tok"] = df["i_norm"].str.split()
    df["jac"] = [jaccard(a, b) for a, b in zip(df["q_tok"], df["i_tok"])]

    # TF-IDF
    corpus = list(df["q_norm"]) + list(df["i_norm"])
    vec = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 2), dtype=np.float32)
    X = vec.fit_transform(corpus)
    n = len(df)
    Qn = normalize(X[:n], norm="l2", copy=False)
    In = normalize(X[n:], norm="l2", copy=False)
    pair_cos = (Qn.multiply(In)).sum(axis=1)
    df["tfidf_cos"] = np.asarray(pair_cos).ravel()

    # pred merge
    if args.pred and os.path.exists(args.pred):
        pred_rows = load_jsonl(args.pred)
        dp = pd.DataFrame([{
            "id": str(r.get("id")),
            "pred": int(r.get("prediction")) if r.get("prediction") is not None else None,
            "prob1": safe_get_prob1(r)
        } for r in pred_rows])
        df = df.merge(dp, on="id", how="left")
        df["label_int"] = df["label"].astype(int)
        df["disagree"] = df["label_int"] != df["pred"]
    else:
        df["pred"] = None
        df["prob1"] = None
        df["disagree"] = False

    # threshold
    thresholds = build_thresholds(df, auto_tune=not args.no_autotune)

    # suspect flag
    df["suspect_with_pred"] = df.apply(lambda r: flag_suspect_row(r, thresholds, use_pred=True), axis=1)
    df["suspect_no_pred"]   = df.apply(lambda r: flag_suspect_row(r, thresholds, use_pred=False), axis=1)

    # 리포트
    cmp = df.groupby("label").agg(
        n=("id","count"),
        sus_with_pred=("suspect_with_pred","sum"),
        sus_no_pred=("suspect_no_pred","sum"),
    ).reset_index()
    cmp["diff"] = cmp["sus_with_pred"] - cmp["sus_no_pred"]
    print("\n[비교: pred 영향 포함 vs 제외]")
    print(cmp.to_string(index=False))

    # suspect 샘플 제외하고 클린 데이터 저장
    clean_df = df[~df["suspect_with_pred"]].copy()
    clean_df[["id","lang","q","i","label"]].to_json(
        args.output, orient="records", lines=True, force_ascii=False
    )

    print(f"✓ Saved refined data: {args.output} ({len(clean_df)} records)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="QI or QC")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output refined JSONL")
    parser.add_argument("--pred", type=str, help="Optional model prediction JSONL")
    parser.add_argument("--no_autotune", action="store_true", help="Disable auto thresholds")
    args = parser.parse_args()

    main(args)
