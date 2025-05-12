#!/usr/bin/env python3
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
import pandas as pd
from tqdm import tqdm
import time

# ——— PREPROCESS FUNCTION ———
def preprocess(ex, args, tokenizer):
    prompt = ex["instruction"].strip() + "\n\n" + ex["input"].strip() + "\n\n"
    resp   = ex["output"].strip()
    pids   = tokenizer(prompt,   add_special_tokens=False).input_ids
    rids   = tokenizer(resp,     add_special_tokens=False).input_ids
    ids     = (pids + rids)[: args.model_max_length]
    # labels = -100 on prompt, actual on response
    labs    = [-100] * len(pids) + rids
    labs    = labs[: len(ids)]
    # pad to max_length
    pad_len = args.model_max_length - len(ids)
    ids    += [tokenizer.pad_token_id] * pad_len
    labs   += [-100] * pad_len
    mask    = [1] * (len(ids) - pad_len) + [0] * pad_len
    return {"input_ids": ids, "attention_mask": mask, "labels": labs}

# train_ds = train_ds.map(preprocess, batched=False, remove_columns=train_ds.column_names)
# eval_ds  = eval_ds.map(preprocess,  batched=False, remove_columns=eval_ds.column_names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", required=True,
                   help="Path to your fine‑tuned LoRA directory")
    p.add_argument("--test_json",   required=True)
    p.add_argument("--output_json", default="predictions.json")
    p.add_argument("--max_new_tokens", type=int, default=80)
    args = p.parse_args()

    # — load adapter config & base model
    cfg = PeftConfig.from_pretrained(args.adapter_path)
    base = cfg.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, padding_side="right", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=torch.float16)
    # merge weights for speed & lower memory
    model = model.merge_and_unload().eval()
    
    df = pd.read_json(args.test_json)
    test_data = df.to_dict("records")
    outputs = []
    latest_file_path = args.output_json.strip(".json")+'_latest.json'

    for ex in tqdm(test_data):
        prompt = ex["instruction"].strip() + "\n\n" + ex["input"].strip() + "\n\n"
        start_time = time.time()
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        gen = model.generate(
            **tokens,
            max_new_tokens = args.max_new_tokens,
            # early_stopping   = True,
            eos_token_id     = tokenizer.eos_token_id
        )
        # crop off prompt tokens
        new_ids = gen[0][ tokens["input_ids"].shape[-1] : ]
        reply   = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        end_time = time.time()
        latency = end_time - start_time

        ex["prediction"] = reply
        ex["latency"] = latency
        outputs.append(ex)
        if len(outputs)%100 == 0:
            print(f"Processed {len(outputs)} examples")
            pd.DataFrame(outputs).to_json(latest_file_path)
            print(f"Saved latest results to {latest_file_path=}")

    pd.DataFrame(outputs).to_json(args.output_json)
if __name__ == "__main__":
    main()

# python talker_predict.py \
#   --adapter_path ~/exp_medmcqa/trt_qwen_2point5_7b_latest \
#   --test_json tr_data/test_mathdial.json \
#   --output_json tr_data/predictions.json
    
# python talker_predict.py   --adapter_path ~/exp_medmcqa/trt_qwen_2point5_7b_latest/checkpoint-150   --test_json tr_data/test_mathdial_extended.json   --output_json tr_data/predictions_qwen_2point5_7b.json 2>&1 | tee logs/test_qwen_talker.log