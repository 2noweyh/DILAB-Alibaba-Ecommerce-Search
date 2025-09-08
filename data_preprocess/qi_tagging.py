# qi_tagging.py
"""
Query–Item 태깅 및 프롬프트 생성 스크립트

- 입력(clean된 QI 데이터 JSONL)과 QC 데이터(JSONL)를 기반으로
  1. 언어 태그([LANG=xx])를 쿼리/아이템에 주입
  2. 타이틀 기반 속성 추출 (brand, color, size, material, category, style)
  3. QC category 연동 (leaf 우선 적용)
  4. 영어 프롬프트 생성 (Llama-3 입력용)
- 최종 산출물에 origin_query_tagged, item_title_tagged 열 포함
"""

import json, re, argparse
from pathlib import Path
from typing import Dict, Optional, List

# -------- Regex / Dicts --------
BRAND_HINTS = [
    r"\bby\s+([A-Za-z0-9\-\_]+)\b",
    r"\bbrand[:\s]+([A-Za-z0-9\-\_]+)\b",
]
BRAND_WHITELIST = {
    "figma","good smile","goodsmile","max factory","bandai","tamashii",
    "shfiguarts","hasbro","mattel","neca","mcfarlane","kotobukiya",
    "lego","tamiya","revell","takara","tomica","hot wheels","hotwheels"
}
EXCLUDED_COLOR_PHRASES = [r"\bred\s+pyramid\s+thing\b"]
COLOR_WORDS = r"(black|white|off[-\s]?white|ivory|cream|gray|grey|blue|navy|green|red|pink|beige|brown|tan|teak|walnut|oak|maple|cherry|mahogany|silver|gold|bronze|clear|transparent)"
SIZE_PAT    = r"(\b\d+([./]\d+)?\s?(cm|mm|in(?:ch)?|ft|ml|l|g|kg|oz)\b|\b(xs|s|m|l|xl|xxl)\b|\b\d+\s?(pcs?|pieces?|pack|set)\b|\b\d+\s?[x×]\s?\d+\s?(cm|mm|in|inch)\b)"
MATERIAL_WORDS = r"(solid wood|engineered wood|mdf|plywood|metal|steel|stainless steel|aluminum|iron|alloy|plastic|pu leather|faux leather|leather|glass|fabric|cotton|linen|polyester|rubber|silicone|rattan|bamboo)"
CATEGORY_WORDS = r"(coffee table set|coffee table|end table|side table|desk|chair|sofa|mat|sheet|sheets|shelf|cabinet|crib|mattress|action figure|model kit|model kits?)"
STYLE_HINTS = [
    r"silent\s*hill\s*2", r"red\s*pyramid\s*thing", r"bubble\s*head\s*nurse",
    r"star\s*wars", r"marvel", r"dc\s*comics", r"pokemon", r"one\s*piece",
    r"gundam", r"naruto", r"dragon\s*ball"
]

# -------- Helpers --------
def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_qc_category_map(qc_path: str) -> Dict[str, str]:
    mp = {}
    p = Path(qc_path)
    if not p.exists():
        return mp
    for d in load_jsonl(qc_path):
        if d.get("task") != "QC":
            continue
        oq = normalize_query(d.get("origin_query", ""))
        cp = d.get("category_path", "")
        if oq and cp:
            mp[oq] = cp
    return mp

def pick_leaf_category(category_path: str) -> Optional[str]:
    parts = [p.strip() for p in category_path.split(",") if p.strip()]
    return parts[-1].lower().replace(" ", "_") if parts else None

def extract_brand(title: str) -> Optional[str]:
    t = title.lower()
    for b in sorted(BRAND_WHITELIST, key=len, reverse=True):
        if re.search(rf"\b{re.escape(b)}\b", t):
            return b.replace(" ", "_")
    for pat in BRAND_HINTS:
        m = re.search(pat, title, flags=re.I)
        if m:
            return m.group(1).strip().lower().replace(" ", "_")
    return None

def extract_color(title: str) -> Optional[str]:
    tl = title.lower()
    for p in EXCLUDED_COLOR_PHRASES:
        if re.search(p, tl):
            return None
    m = re.search(COLOR_WORDS, title, flags=re.I)
    return m.group(1).lower() if m else None

def extract_size(title: str) -> Optional[str]:
    m = re.search(SIZE_PAT, title, flags=re.I)
    return m.group(0).lower() if m else None

def extract_material(title: str) -> Optional[str]:
    m = re.search(MATERIAL_WORDS, title, flags=re.I)
    return m.group(1).lower() if m else None

def extract_category_from_title(title: str) -> Optional[str]:
    m = re.search(CATEGORY_WORDS, title, flags=re.I)
    if m:
        return m.group(0).lower().replace(" ", "_")
    if "table" in title.lower():
        return "table"
    return None

def extract_styles(title: str) -> List[str]:
    tl = title.lower()
    out = []
    for pat in STYLE_HINTS:
        m = re.search(pat, tl, flags=re.I)
        if m:
            out.append(re.sub(r"\s+", "_", m.group(0).lower()))
    # dedup
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

def safe(v):
    return v if v else "unknown"

# -------- Prompt builder (English only; add [LANG=xx]) --------
SYS_GUIDE_EN = "You are a precise data labeler and copywriter for an e-commerce catalog."
INSTR_EN = (
    "Given the user query and the product title/attributes, output:\n"
    "1) inline tags with a fixed, non-nested tag set;\n"
    "2) a concise English description;\n"
    "3) a binary relevance label (1 if the item fully matches the query on product type, brand, model, and key attributes; otherwise 0)."
)
FORMAT_RULES = (
    "Use only these tags (lowercase, no nesting, words joined by underscores if needed):\n"
    "[cat], [mat], [color], [size], [style], [feature], [audience], [brand]\n"
    "Output format (strictly):\n"
    "[TAGS] ... [/TAGS]\n"
    "[DESC] ... [/DESC]\n"
    "[LABEL] 0 or 1 [/LABEL]"
)
CONSTRAINTS = (
    "- Use only verifiable info from the title/attributes/category context.\n"
    "- Do not mention unknown attributes.\n"
    "- Be objective and concise. Description should be 1–2 sentences.\n"
    "- Respond in English.\n"
    "- Label 1 only if the item corresponds to the query in product type, brand, model, and salient attributes; else 0."
)

def build_prompt_en(origin_query: str, item_title: str, attrs: Dict[str, str], qc_leaf: Optional[str], lang_code: str):
    fields = []
    for k in ["brand", "category", "material", "color", "size"]:
        v = attrs.get(k)
        if v and v != "unknown":
            fields.append(f"{k}: {v}")
    styles = attrs.get("style_list") or []
    if styles:
        fields.append("style: " + "|".join(styles))
    if qc_leaf:
        fields.append(f"qc_leaf: {qc_leaf}")
    attributes_str = ", ".join(fields) if fields else "N/A"

    origin_query_tagged = f"[LANG={lang_code}] {origin_query}".strip()
    item_title_tagged   = f"{item_title}".strip()
    if attributes_str and attributes_str != "N/A":
        item_title_tagged += f" [ATTR] {attributes_str}"

    prompt = f"""{SYS_GUIDE_EN}

{INSTR_EN}
{FORMAT_RULES}

[INPUT]
[LANG={lang_code}]
query: {origin_query}
title: {item_title}
attributes: {attributes_str}

[CONSTRAINTS]
{CONSTRAINTS}
""".strip()

    return prompt, attributes_str, origin_query_tagged, item_title_tagged

# -------- Runner --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Clean QI dataset JSONL (with origin_query/item_title/language/label)")
    parser.add_argument("--qc_path", type=str, required=True,
                        help="QC dataset JSONL for category mapping")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL with prompts and tagged fields")
    args = parser.parse_args()

    qc_map = load_qc_category_map(args.qc_path)
    cnt = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)

            origin_query = d.get("origin_query", d.get("q", "")) or ""
            title        = (d.get("item_title", d.get("i", "")) or "").strip()
            lang_code    = (d.get("language", d.get("lang", "unk")) or "unk").strip()
            label        = d.get("label")

            oq_norm = normalize_query(origin_query)


            # 타이틀 기반 속성 추출
            styles = extract_styles(title)
            attrs = {
                "brand": safe(extract_brand(title)),
                "color": safe(extract_color(title)),
                "size": safe(extract_size(title)),
                "material": safe(extract_material(title)),
                "category": safe(extract_category_from_title(title)),
                "style_list": styles,
            }

            # QC leaf override
            qc_leaf = None
            if oq_norm in qc_map:
                qc_leaf = pick_leaf_category(qc_map[oq_norm])
                if qc_leaf:
                    attrs["category"] = qc_leaf

            prompt_for_train, attributes_str, q_tagged, i_tagged = build_prompt_en(
                origin_query, title, attrs, qc_leaf, lang_code
            )

            out = {
                "id": d.get("id"),
                "task": d.get("task", "QI"),
                "language": lang_code,
                "origin_query": origin_query,
                "item_title": title,
                "origin_query_tagged": q_tagged,
                "item_title_tagged":   i_tagged,
                "label": label,
                "attributes_extracted": attrs,
                "attributes_str": attributes_str,
                "prompt_for_train": prompt_for_train
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            cnt += 1

    print(f"✓ Wrote {cnt} records to {args.output}")


if __name__ == "__main__":
    main()
