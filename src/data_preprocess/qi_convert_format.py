import json
import os
import argparse
from tqdm import tqdm

# In-context learning
qi_icl_example = """You are given a user query and a product title.
Decide if the product is relevant to the query.
Relevant: matches user intent (category/type match, brand/specs may differ).
Not relevant: unrelated type or accessory instead of main item.
Respond only with "yes" or "no".

Query: [LANG=en] wireless bluetooth earbuds
Product: Sony WF-1000XM5 Noise Cancelling Headphones
Answer: yes

Query: [LANG=ko] 유리컵
Product: Wireless Bluetooth Earbuds
Answer: no
"""

def convert_qi_format_with_icl(input_path, output_path, split="train"):
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Converting"):
            row = json.loads(line)

            instruction = (
                "You are given a user query and a product title.\n"
                "Decide if the product is relevant to the query.\n"
                "Relevant: matches user intent (category/type match, brand/specs may differ).\n"
                "Not relevant: unrelated type or accessory instead of main item.\n"
                "Respond only with \"yes\" or \"no\"."
            )

            query = row.get("origin_query_tagged", row["origin_query"])
            title = row.get("item_title_tagged", row["item_title"])

            gen = row.get("generation", {})
            gen_tags = gen.get("tags_block") or ""
            gen_desc = gen.get("desc") or ""
            gen_label = gen.get("pred_label")

            input_text = f"Query: {query}\nProduct: {title}"
            if gen_tags or gen_desc:
                input_text += f"\n\n[GEN_TAGS] {gen_tags}\n[GEN_DESC] {gen_desc}"

            output = None
            if split == "train":
                output = "yes" if str(row.get("gold_label", "")) == "1" else "no"
            else:
                output = ""

            data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "options": qi_icl_example,
                "language": row.get("language", "unknown"),
                "gen_label": gen_label,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(data)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL with generations")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file (ECInstruct format)")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train", help="Dataset split")
    args = parser.parse_args()

    convert_qi_format_with_icl(
        input_path=args.input,
        output_path=args.output,
        split=args.split
    )

if __name__ == "__main__":
    main()