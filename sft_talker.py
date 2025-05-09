import unsloth
import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
)
from unsloth import FastLanguageModel
import torch
import csv
import wandb
import logging
from datetime import datetime
from peft import LoraConfig, TaskType

# ---------------- ARGUMENT PARSING ---------------- #
parser = argparse.ArgumentParser(description="Fine-tune with Unsloth on talker using flexible hyperparameters")
parser.add_argument("--model_name", type=str, required=True, help="Model name or path (e.g., Qwen/Qwen2.5-1.5B-Instruct)")
parser.add_argument("--data_path", type=str, required=True, help="Path to training data (pickle)")
parser.add_argument("--val_path", type=str, default=None, help="Path to validation data (pickle), defaults to training data")
parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for model downloads")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs and logs")
parser.add_argument("--num_train_samples", type=int, default=None, help="Number of train samples (None = full)")
parser.add_argument("--model_max_length", type=int, default=512, help="Max sequence length for tokenizer/model")
parser.add_argument("--low_rank_training", action="store_true", help="Enable 4-bit / LoRA training mode")
parser.add_argument("--bf16", action="store_true", help="Use bf16 if CUDA supports it")
parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmuls on Ampere+ GPUs")
parser.add_argument("--ddp_find_unused_parameters", type=bool, default=False, help="DDP setting to find unused parameters")
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--save_strategy", type=str, default="steps", help="Checkpoint save strategy: steps or epoch")
parser.add_argument("--save_total_limit", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warmup_ratio", type=float, default=0.0)
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--logging_steps", type=int, default=20)
args = parser.parse_args()

# ---------------- PATH SETUP ---------------- #
os.makedirs(args.output_dir, exist_ok=True)
train_path = args.data_path
val_path = args.val_path or args.data_path
metrics_csv = os.path.join(args.output_dir, "metrics_log.csv")

# ---------------- LOGGER ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Starting training: {args.model_name}")
logger.info(f"Train data: {train_path}, Val data: {val_path}")

# ---------------- LOAD DATA ---------------- #

from sklearn.model_selection import train_test_split

# train_df = pd.read_pickle(train_path)
# val_df = pd.read_pickle(val_path)
# ---------------- LOAD SFT DATA ---------------- #

if args.data_path:
    sft_df = pd.read_pickle(args.data_path)
else:
    # default sft data path
    sft_df = pd.read_pickle("tr_data/mathdial_sft.pkl")

# 80/20 train‑val split
train_df, val_df = train_test_split(sft_df, test_size=0.2, random_state=42)

# If requested, subsample *only* the train set
if args.num_train_samples:
    train_df = train_df.sample(n=args.num_train_samples, random_state=42)


# ---------------- FORMAT INSTRUCTION ---------------- #
def format_instruction(row):
    """
    Given a row with keys:
      - instruction: the static prompt (talker_prompt.txt contents)
      - input: JSON string with persona, reasoner_context, conversation_history
      - output: the teacher's response
    Return the single 'text' field = instruction + input + output
    """
    text = (
        f"{row['instruction'].strip()}\n\n"
        f"{row['input'].strip()}\n\n"
        f"{row['output'].strip()}"
    )
    return {"text": text}

# Apply to train/val
train_list = train_df.apply(format_instruction, axis=1).tolist()
val_list   = val_df.apply(format_instruction, axis=1).tolist()

# Build HuggingFace Datasets
from datasets import Dataset
train_dataset = Dataset.from_list(train_list)
val_dataset   = Dataset.from_list(val_list)

# ---------------- LOAD MODEL ---------------- #
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    cache_dir=args.cache_dir,
    max_seq_length=args.model_max_length,
    dtype=torch.bfloat16 if args.bf16 else torch.float16,
    load_in_4bit=args.low_rank_training,
)
FastLanguageModel.for_training(model)

lora_cfg = LoraConfig(
    target_modules=["q_proj","v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
model = FastLanguageModel.get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ---------------- TOKENIZE ---------------- #
def tokenize(example):
    enc = tokenizer(
        example["text"],
        max_length=args.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    return {k: v.squeeze() for k,v in enc.items()}

train_dataset = train_dataset.map(tokenize, remove_columns=["text"])
val_dataset   = val_dataset.map(tokenize, remove_columns=["text"])

# ---------------- CSV LOGGER CALLBACK ---------------- #
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs: return
        logs["step"] = state.global_step
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs.keys()))
            if write_header: writer.writeheader()
            writer.writerow(logs)

# ---------------- TF32 ---------------- #
if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

# ---------------- TRAINING ARGS ---------------- #
training_args = TrainingArguments(
    output_dir=args.output_dir,
    ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    evaluation_strategy=args.save_strategy,
    save_strategy=args.save_strategy,
    save_total_limit=args.save_total_limit,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler_type,
    bf16=args.bf16 and torch.cuda.is_bf16_supported(),
    fp16=(not args.bf16) and torch.cuda.is_available(),
    logging_dir=os.path.join(args.output_dir, "logs"),
    logging_steps=args.logging_steps,
    report_to="wandb",
    run_name=f"{args.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# ---------------- WANDB INIT ---------------- #
wandb.init(project="unsloth-talker", name=training_args.run_name)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[CSVLoggerCallback(metrics_csv)]
)

# ---------------- TRAIN ---------------- #
try:
    trainer.train()
except Exception as e:
    logger.error(f"Training failed: {e}")
    logger.info("You can resume with --resume_from_checkpoint <path>")
finally:
    wandb.finish()
    logger.info(f"Finished training for {args.model_name}")
