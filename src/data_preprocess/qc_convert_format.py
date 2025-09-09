import json
import os
import argparse
from tqdm import tqdm

# In-context learning
qc_icl_example = """Decide if the category is relevant to the query.
Relevant: matches user intent (correct category).
Not relevant: unrelated or wrong category.
Respond only with "yes" or "no".

Query: [LANG=es] por watch
Category: [D1] watches [/D1] [D2] pocket & fob watches [/D2]
Answer: yes

Query: [LANG=ko] 베이스젤
Category: [D1] beauty & health [/D1] [D2] nail art & tools [/D2] [D3] nail art [/D3] [D4] top coat [/D4]
Answer: no
"""

def convert_qc_format_with_icl(jsonl_path, output_path, split="train"):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Converting"):
            row = json.loads(line)

            instruction = (
                "You are given a user query and a product category.\n"
                "Decide if the category is relevant to the query.\n"
                "Relevant: matches user intent (correct category).\n"
                "Not relevant: unrelated or wrong category.\n"
                "Respond only with \"yes\" or \"no\"."
            )

            query = row.get("origin_query_tagged", row["origin_query"])
            category = row.get("category_path_tagged", row["category_path"])

            gen = row.get("generation", {})
            gen_tags = gen.get("tags_block") or ""
            gen_desc = gen.get("desc") or ""
            gen_label = gen.get("pred_label")

            input_text = f"Query: {query}\nCategory: {category}"
            if gen_tags or gen_desc:
                input_text += f"\n\n[GEN_TAGS] {gen_tags}\n[GEN_DESC] {gen_desc}"

            output = None
            if split == "train": 
                output = "yes" if str(row.get("label", "")) == "1" else "no"
            else:
                output = ""

            data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "options": qc_icl_example,
                "language": row.get("language", "unknown"),
                "gen_label": gen_label,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(data)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL with generations")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON file (ECInstruct format)")
    parser.add_argument("--split", type=str, choices=["train", "dev"], default="train", help="Dataset split")
    args = parser.parse_args()

    convert_qc_format_with_icl(
        jsonl_path=args.input_path,
        output_path=args.output_path,
        split=args.split
    )

if __name__ == "__main__":
    main()