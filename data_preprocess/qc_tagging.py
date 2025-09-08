# ./data_preprocess/qc_tagging.py

import argparse
import json
from tqdm import tqdm

def qc_tagging(input_file, output_file): 
    ''' Preprocess the input JSONL file for QC task.
        Adds language tags(ex. [LANG=en]) to queries and depth tags(ex. [D1][/D1]) to category paths.
    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to save the preprocessed JSONL file.
    Returns:
        str: Path to the preprocessed JSONL file.
    '''

    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        lines = f.readlines()

        for line in tqdm(lines, total=len(lines)):
            item = json.loads(line.strip())

            # Tagging language to query
            lang = f"[LANG={item['language']}] "
            item["origin_query"] = lang + item["origin_query"]

            # Tagging depth to category paths
            category = ""
            for i, cate in enumerate(item["category_path"].split(","), 1):
                temp = f"[D{i}] {cate.strip()} [/D{i}]"
                category += temp + " "
            item["category_path"] = category.strip()

            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Complete QC Preprocessing! Saved to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QC Preprocessing : Tagging language and depth")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    qc_tagging(args.input, args.output)
