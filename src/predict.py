import os
import sys
import time
import pandas as pd

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import json
import torch.nn.functional as F
import numpy as np
from prompter import Prompter
import gc
import pdb
from tqdm import tqdm
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

@torch.no_grad()
def score_yes_no_batch(model, tokenizer, prompts, yes_token="yes", no_token="no"):
    results = []
    device = next(model.parameters()).device

    yes_inputs = tokenizer(
        [p + yes_token for p in prompts],
        return_tensors="pt", padding=True, truncation=True
    )
    no_inputs = tokenizer(
        [p + no_token for p in prompts],
        return_tensors="pt", padding=True, truncation=True
    )

    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    def make_labels(input_ids):
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        return labels

    yes_inputs = {k: v.to(device) for k, v in yes_inputs.items()}
    yes_labels = make_labels(yes_inputs["input_ids"])
    yes_out = model(**yes_inputs)
    yes_logits = yes_out.logits[:, :-1, :]               # shift
    yes_target = yes_labels[:, 1:]                       # shift
    yes_mask = (yes_target != -100)
    yes_logprob = (-F.cross_entropy(
        yes_logits[yes_mask], yes_target[yes_mask], reduction="sum"
    )).item() if yes_mask.any() else float("-inf")

    no_inputs = {k: v.to(device) for k, v in no_inputs.items()}
    no_labels = make_labels(no_inputs["input_ids"])
    no_out = model(**no_inputs)
    no_logits = no_out.logits[:, :-1, :]
    no_target = no_labels[:, 1:]
    no_mask = (no_target != -100)
    no_logprob = (-F.cross_entropy(
        no_logits[no_mask], no_target[no_mask], reduction="sum"
    )).item() if no_mask.any() else float("-inf")


    results = []
    for b in range(len(prompts)):
        res = {}
        for candidate, key in [(yes_token, "yes"), (no_token, "no")]:
            inputs = tokenizer(prompts[b] + candidate, return_tensors="pt").to(device)
            labels = inputs["input_ids"].clone()
            labels[:, :len(tokenizer(prompts[b])["input_ids"])] = -100
            out = model(**inputs)
            logits = out.logits[:, :-1, :]
            target = labels[:, 1:]
            mask = (target != -100)
            logprob = (-F.cross_entropy(
                logits[mask], target[mask], reduction="sum"
            ).item()) if mask.any() else float("-inf")
            res[f"logprob_{key}"] = logprob

        lp_yes, lp_no = res["logprob_yes"], res["logprob_no"]

        m = max(lp_yes, lp_no)
        p_yes = np.exp(lp_yes - m) / (np.exp(lp_yes - m) + np.exp(lp_no - m))
        res["prob1"] = float(p_yes)
        results.append(res)
    return results

def main(
    load_8bit: bool = False,
    use_lora: bool = True,
    base_model: str = "../llama30B_hf",
    lora_weights: str = "",
    prompt_template: str = "mistral",
    data_path: str = "",
    task: str = "",
    setting: str = "",
    output_data_path: str = "",
    save_scores: bool = False,     

):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    prompter = Prompter(prompt_template)
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":

        if base_model not in ['microsoft/phi-2', 'NingLab/eCeLLM-S']:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map={"": 0},
                # trust_remote_code=True,
                # attn_implementation='flash_attention_2',
                attn_implementation='eager',
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        elif base_model in ['mistralai/Mistral-7B-Instruct-v0.2']:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map={"": 0},
                attn_implementation='eager',
                # trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map={"": 0},
                # trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)

        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )

    if not load_8bit:
        model.half()

    model.eval()
    #if torch.__version__ >= "2" and sys.platform != "win32":
        # model = torch.compile(model)

    if not model.config.eos_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        model.config.eos_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = 'left'

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.float16, 
        device_map="auto",
    )

    dataset = load_dataset("json", data_files=data_path)['train']

    instructions, inputs, options, ids = [], [], [], []

    for data in dataset:
        instructions.append(data["instruction"])
        inputs.append(data["input"])
        options.append(data.get("options", None))
        ids.append(data["id"] if "id" in data else str(len(ids)))

    skipped_ids = set()
    if os.path.exists(output_data_path):
        backup_path = output_data_path + ".bak"
        print(f"[INFO] Found existing output. Backing up to {backup_path}")
        shutil.move(output_data_path, backup_path)

        with open(backup_path, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    skipped_ids.add(str(example["id"]))
                except:
                    continue

    print(f"[INFO] Skipping {len(skipped_ids)} previously processed examples.")

    output_mode = "w"
    max_batch_size = 2
    for i in tqdm(range(0, len(instructions), max_batch_size), desc="Running batches"):
        instruction_batch = instructions[i:i + max_batch_size]
        input_batch = inputs[i:i + max_batch_size]
        options_batch = options[i:i + max_batch_size]
        ids_batch = ids[i:i + max_batch_size]

        # === Skip already done ===
        filtered_batch = [
            (inst, inp, opt, idx) for inst, inp, opt, idx in zip(instruction_batch, input_batch, options_batch, ids_batch)
            if str(idx) not in skipped_ids
        ]

        if len(filtered_batch) == 0:
            continue 

        prompts = [prompter.generate_prompt(inst, inp, opt) for inst, inp, opt, _ in filtered_batch]
        batch_results = evaluate(prompter, prompts, tokenizer, pipe, len(filtered_batch))
    
        scores = None
        if save_scores:
            scores = score_yes_no_batch(model, tokenizer, prompts, yes_token="yes", no_token="no")

        with open(output_data_path, output_mode, encoding='utf-8') as f:
            for idx_in_batch, ((_, _, _, idx), response) in enumerate(zip(filtered_batch, batch_results)):
                pred = 1 if response.strip().lower() == "yes" else 0
                result = {
                    "id": int(idx)+1,
                    "prediction": pred
                }
                if save_scores and scores is not None:
                    result.update({
                        "prob1": round(float(scores[idx_in_batch]["prob1"]), 6),
                        "logprob_yes": round(float(scores[idx_in_batch]["logprob_yes"]), 3),
                        "logprob_no":  round(float(scores[idx_in_batch]["logprob_no"]),  3),
                    })
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


        output_mode = "a"
        gc.collect()
        torch.cuda.empty_cache()


def extract_query(input_text):
    for line in input_text.split("\n"):
        if line.lower().startswith("query:"):
            return line.split(":", 1)[1].strip()
    return ""

def extract_product(input_text):
    for line in input_text.split("\n"):
        if line.lower().startswith("product:"):
            return line.split(":", 1)[1].strip()
    return ""


def evaluate(prompter, prompts, tokenizer, pipe, batch_size):
    batch_outputs = []

    generation_output = pipe(
        prompts,
        do_sample=True,
        max_new_tokens=5,
        temperature=0.15,
        top_p=0.95,
        num_return_sequences=1,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, 
        batch_size=batch_size,
    )

    for i in range(len(generation_output)):    
        resp = prompter.get_response(generation_output[i][0]['generated_text'])
        batch_outputs.append(resp)

    return batch_outputs


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)

