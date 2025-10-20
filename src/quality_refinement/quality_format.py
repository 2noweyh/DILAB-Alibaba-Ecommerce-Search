import json
import os
import argparse

qc_icl_example = """You are given a user query and a product category path.
Determine whether the category is appropriate for the query.
Respond only with "yes" or "no".

Query: 1/14 rc volvo fh16
Category Path: toys & hobbies,remote control toys,rc trucks

Answer: yes

Query: 미니핫도그
Category Path: home & garden,home textile,carpet

Answer: no

"""

def convert_qc_format_with_icl(jsonl_path, output_path, has_label=True):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            instruction = "You are given a user query and a product category path.\nDetermine whether the category is appropriate for the query.\nRespond only with \"yes\" or \"no\"."
            input_text = f"Query: {row['origin_query']}\nCategory Path: {row['category_path']}"
            output = "yes" if str(row.get("label", "")) == "1" else "no" if has_label else None
            # if has_label:
            #     output = "yes" if str(row.get("label", "")) == "1" else "no"
            # else:
            #     output = ""

            data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output if has_label else "",
                "options": qc_icl_example,
                "language": row.get("language", "unknown")
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=4, ensure_ascii=False)

    print(f"✅ Converted {len(data)} samples to {output_path}")

qi_icl_example = """You are given a user query and a product title.
Decide if the product is relevant to the query.
Respond only with "yes" or "no".

Query: wireless bluetooth earbuds
Product: Sony WF-1000XM5 Noise Cancelling Headphones

Answer: yes

Query: leather handbag
Product: Wireless Bluetooth Earbuds

Answer: no

"""

def convert_qi_format_with_icl(jsonl_path, output_path, has_label=True):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            instruction = "You are given a user query and a product title.\nDecide if the product is relevant to the query.\nRespond only with \"yes\" or \"no\"."
            input_text = f"Query: {row['origin_query']}\nProduct: {row['item_title']}"
            # input_text = f"Query: {row['query_tagged']}\nProduct: {row['title_tagged']}"
            output = "yes" if str(row.get("label", "")) == "1" else "no" if has_label else None

            data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output if has_label else "",
                "options": qi_icl_example,
                "language": row.get("language", "unknown")
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=4, ensure_ascii=False)

    print(f"✅ Converted {len(data)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, choices=[
        "convert_qi_format_with_icl", "convert_qc_format_with_icl"
    ])
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--has_label", action="store_true", default=True)
    args = parser.parse_args()

    if args.run == "convert_qi_format_with_icl":
        convert_qi_format_with_icl(args.input, args.output, has_label=args.has_label)
    elif args.run == "convert_qc_format_with_icl":
        convert_qc_format_with_icl(args.input, args.output, has_label=args.has_label)