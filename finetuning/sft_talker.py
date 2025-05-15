#!/usr/bin/env python3
import unsloth
import os
import argparse
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import wandb
from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, TaskType
from unsloth import FastLanguageModel

import os, torch

# # In a torchrun ddp launch, LOCAL_RANK tells you which GPU this process owns
# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# torch.cuda.set_device(local_rank)


# ——— ARGPARSE ———
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",    required=True)
parser.add_argument("--data_path",     required=True)
parser.add_argument("--val_path",      default=None)
parser.add_argument("--cache_dir",     default=None)
parser.add_argument("--output_dir",    required=True)
parser.add_argument("--model_max_length", type=int, default=512)
parser.add_argument("--num_train_epochs",   type=int, default=3)
parser.add_argument("--per_device_train_batch_size",  type=int, default=4)
parser.add_argument("--per_device_eval_batch_size",   type=int, default=4)
parser.add_argument("--gradient_accumulation_steps",  type=int, default=1)
parser.add_argument("--learning_rate",    type=float, default=5e-5)
parser.add_argument("--weight_decay",     type=float, default=0.0)
parser.add_argument("--warmup_ratio",     type=float, default=0.0)
parser.add_argument("--lr_scheduler_type",     default="linear")
parser.add_argument("--logging_steps",    type=int, default=20)
parser.add_argument("--save_strategy",    default="steps")
parser.add_argument("--save_steps",       type=int, default=500)
parser.add_argument("--save_total_limit", type=int, default=2)
parser.add_argument("--ddp_find_unused_parameters", type=bool, default=False)
parser.add_argument("--low_rank_training", action="store_true", help="Enable 4-bit / LoRA training mode")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--tf32", action="store_true")
parser.add_argument("--deepspeed", default="/home/varshinibala_umass_edu/medmcqa/deepspeed/config.json")
parser.add_argument("--resume_from_checkpoint",default=None,help="Path to a HuggingFace checkpoint folder (e.g. checkpoint-50) to resume training from")

args = parser.parse_args()

# ——— LOGGING & DIRS ———
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"Training {args.model_name} → {args.output_dir}")

# ——— LOAD DATAFRAME ———
# assert args.data_path.endswith(".json")
if args.data_path:
    df = pd.read_json(args.data_path)
else:
    df = pd.read_json("tr_data/train_mathdial.json")


# train and validation with 80-20 splits
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ——— DATASET → HF Dataset ———
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_ds  = Dataset.from_pandas(val_df.reset_index(drop=True))

# ——— LOAD MODEL & TOKENIZER ———
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = args.model_name,
    cache_dir     = args.cache_dir,
    max_seq_length= args.model_max_length,
    dtype         = torch.bfloat16 if args.bf16 else torch.float16,
    # load_in_4bit  = args.low_rank_training, #false
)
FastLanguageModel.for_training(model)

# ——— ATTACH LoRA ———
lora_cfg = LoraConfig(
    target_modules = ["q_proj","v_proj"],
    task_type       = TaskType.CAUSAL_LM,
    lora_alpha      = 32,
    lora_dropout    = 0.05,
)
model = FastLanguageModel.get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ——— PREPROCESS FUNCTION ———
def preprocess(ex):
    prompt = ex["input_prompt"].strip() + "\n\n"
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

train_ds = train_ds.map(preprocess, batched=False, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(preprocess,  batched=False, remove_columns=eval_ds.column_names)

# ——— TRAINING ARGS ———
if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

training_args = TrainingArguments(
    output_dir                  = args.output_dir,
    ddp_find_unused_parameters  = args.ddp_find_unused_parameters,
    num_train_epochs            = args.num_train_epochs,
    per_device_train_batch_size = args.per_device_train_batch_size,
    per_device_eval_batch_size  = args.per_device_eval_batch_size,
    gradient_accumulation_steps = args.gradient_accumulation_steps,
    learning_rate               = args.learning_rate,
    weight_decay                = args.weight_decay,
    warmup_ratio                = args.warmup_ratio,
    lr_scheduler_type           = args.lr_scheduler_type,
    bf16                        = args.bf16 and torch.cuda.is_bf16_supported(),
    fp16                        = (not args.bf16) and torch.cuda.is_available(),
    evaluation_strategy         = args.save_strategy,
    save_strategy               = args.save_strategy,
    save_steps                  = args.save_steps,
    save_total_limit            = args.save_total_limit,
    logging_steps               = args.logging_steps,
    logging_dir                 = os.path.join(args.output_dir, "logs"),
    report_to                   = "wandb",
    run_name                    = f"{os.path.basename(args.output_dir)}_{datetime.now():%Y%m%d_%H%M%S}",
    deepspeed                   = args.deepspeed
)

# ——— WANDB & TRAINER ———
wandb.init(project="unsloth-talker-updated", name=training_args.run_name)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = eval_ds,
    tokenizer       = tokenizer,
    data_collator  = default_data_collator,
)
# trainer.train()

if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()
# trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
wandb.finish()

