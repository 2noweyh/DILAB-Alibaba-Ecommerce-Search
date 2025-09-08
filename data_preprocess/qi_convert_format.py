import json
import os
import argparse
from tqdm import tqdm

# In-context learning 예시 (2개만 유지)
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

def convert_qi_format_with_icl(jsonl_path, output_path, split="train"):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
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
    parser.add_argument("--jsonl_path", type=str, required=True, help="Input JSONL with generations")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON file (ECInstruct format)")
    parser.add_argument("--split", type=str, choices=["train", "dev"], default="train", help="Dataset split")
    args = parser.parse_args()

    convert_qi_format_with_icl(
        input_path=args.input,
        output_path=args.output,
        split=args.split
    )

if __name__ == "__main__":
    main()

'''
python convert_qi_format.py \
  --jsonl_path ./outputs/train_generations_vllm_v2.jsonl \
  --split train \
  --output_path ./outputs/train_QI_ecellm_v2.txt \

python convert_qi_format.py \
  --jsonl_path ./outputs/dev_generations_vllm_v2.jsonl \
  --split train \
  --output_path ./outputs/dev_QI_ecellm_v2.txt \
'''

'''
OUTPUT 예시
{
  "instruction": "You are given a user query and a product title.\nDecide if the product is relevant to the query.\nRelevant: matches user intent (category/type match, brand/specs may differ).\nNot relevant: unrelated type or accessory instead of main item.\nRespond only with \"yes\" or \"no\".",
  "input": "Query: [LANG=en] nighty\nProduct: Womens Pajama Sets Lace Shorts Nighty Sleepwear\n\n[GEN_TAGS] nighty_cat\n[GEN_DESC] A women's pajama set with lace shorts and solid underwear for a comfortable and sexy sleepwear experience.",
  "output": "yes",
  "options": "You are given a user query and a product title.\nDecide if the product is relevant to the query.\nRelevant: matches user intent, even if brand/specs differ.\nNot relevant: different type or unrelated.\nRespond only with \"yes\" or \"no\".\n\nQuery: wireless bluetooth earbuds\nProduct: Sony WF-1000XM5 Noise Cancelling Headphones\nAnswer: yes\n\nQuery: leather handbag\nProduct: Wireless Bluetooth Earbuds\nAnswer: no",
  "language": "en",
  "gen_label": "1"
}
'''