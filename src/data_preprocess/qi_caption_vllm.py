import argparse
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# ---------------- Parser ----------------
TAGS_RE  = re.compile(r"\[TAGS\](.+?)\[/TAGS\]", re.S | re.I)
DESC_RE  = re.compile(r"\[DESC\](.+?)\[/DESC\]", re.S | re.I)
LABEL_RE = re.compile(r"\[LABEL\]\s*([01])\s*\[/LABEL\]", re.S | re.I)

def parse_generation(gen_text: str):
    def grab(rx):
        m = rx.search(gen_text)
        return m.group(1).strip() if m else None
    out = {
        "tags_block": grab(TAGS_RE),
        "desc":       grab(DESC_RE),
        "pred_label": grab(LABEL_RE),
        "raw":        gen_text.strip(),
    }
    out["is_valid"] = (
        out["tags_block"] is not None and 
        out["desc"] is not None and 
        out["pred_label"] in {"0","1"}
    )
    return out

# ---------------- Runner ----------------
def run_vllm(in_path: str, out_path: str,
             limit: int = -1,
             max_new_tokens: int = 256,
             temperature: float = 0.2,
             top_p: float = 0.9,
             batch_size: int = 64):
    llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.4)


    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens
    )

    records, prompts = [], []
    with open(in_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            if limit >= 0 and len(records) >= limit:
                break
            d = json.loads(line)
            prompt = d.get("prompt_for_train") or d.get("prompt_for_desc")
            if not prompt:
                continue
            records.append(d)
            prompts.append(prompt)

    print(f"✓ Loaded {len(records)} records. Generating...")

    outputs = llm.generate(prompts, sampling_params)

    with open(out_path, "w", encoding="utf-8") as fout:
        for rec, out in tqdm(zip(records, outputs), total=len(records), desc="Saving"):
            text = out.outputs[0].text.strip()
            parsed = parse_generation(text)

            out_rec = {
                "id": rec.get("id"),
                "task": rec.get("task"),
                "language": rec.get("language"),
                "origin_query": rec.get("origin_query"),
                "item_title": rec.get("item_title"),
                "origin_query_tagged": rec.get("origin_query_tagged"),
                "item_title_tagged":   rec.get("item_title_tagged"),
                "gold_label": rec.get("label"),
                "attributes_extracted": rec.get("attributes_extracted"),
                "attributes_str": rec.get("attributes_str"),
                "prompt_for_train": prompt,
                "generation": parsed,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"✓ Generated {len(records)} records -> {out_path}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  dest="in_path",
                    default="./outputs/dev_QI_with_prompts.jsonl")
    ap.add_argument("--output", dest="out_path",
                    default="./outputs/dev_generations_vllm.jsonl")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="dummy arg (vllm handles batching internally)")
    args = ap.parse_args()

    run_vllm(
        in_path=args.in_path,
        out_path=args.out_path,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()