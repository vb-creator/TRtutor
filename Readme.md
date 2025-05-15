# TRTutor: Talker-Reasoner Math Tutor

This repository contains code to train and deploy a Socratic 'talker-reasoner' framework to implement a math tutor. It includes,

* **Dataset Generation**: Scripts to generate training dataset with Talker-reasoner enhanced dialogues from raw mathdial conversation samples.
* **Reasoner Context Generation**: Code to generate chain of thought, belief state, and final answers for each problem using SOTA reasoner models like o3-mini, deepseek-r1.
* **Talker Fine‑tuning**: `finetuning_sft_talker.py` to run supervised fine‑tuning with LoRA adapters.
* **Inference**: `inference/talker_predict.py` to generate tutor responses at inference time.

---

<!-- ## Repository Structure

```text
├── data/
│   ├── raw/                   # raw problem definitions and student solutions
│   ├── tr_data/               # processed JSON datasets for train/test
│   │   ├── train_mathdial.json
│   │   └── test_mathdial.json
│   └── cache/                 # tokenizer and model cache
│
├── scripts/
│   ├── dataset_generation.py         # generate SFT examples from raw data
│   ├── reasoner_context_generation.py # compute chain_of_thought, belief_state, final_answer
│
├── finetuning/
│   └── sft_talker.py          # supervised fine‑tuning script using Unsloth + PEFT + Deepspeed
│
├── inference/
│   └── talker_predict.py      # inference script to load adapter and generate responses
│
├── talker_prompt.txt          # static instruction template for the talker
├── reasoner_prompt.txt        # static instruction template for the reasoner
└── README.md                  # this file
```

--- -->

## Installation

1. Create a Python environment (poetry) with Python 3.9+.
2. Install dependencies (first install poetry)

   ```bash
   poetry install
   ```
3. Configure W&B to track model results (optional):

   ```bash
   wandb login
   export WANDB_PROJECT="talker-tutor"
   ```

---

## Dataset Generation

Use `dataset_generation/enhanced_dialogue_generation.py` to generate reasoner_context and personalized conversation dataset using openAI models.

```bash
python dataset_generation/enhanced_dialogue_generation.py \
  --input_dir data/mathdial_df.pkl \
  --output_json tr_data/train_mathdial.json
```

Use `dataset_generation/prepare_sft_data.py` to format the dataset for SFT,

* `instruction`: the talker prompt template
* `input`: JSON with `student_persona`, `reasoner_context`, `conversation_history`
* `output`: the teacher’s next response

---

## Reasoner Context Generation

Compute chain of thought, belief state, and final answer using OpenAI (API key), Deepseek-R1 (using Fireworks API key):

```bash
python scripts/reasoner_context_generation.py \
  --problems data/raw/problems.json \
  --solutions data/raw/solutions.json \
  --output_json data/tr_data/train_mathdial.json
```

This enriches talker's shared context using `reasoner_context`.

---

## Talker Fine‑tuning

Run `finetuning/sft_talker.py` with `torchrun` below command requires access to 2 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 finetuning/sft_talker.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --bf16 \
  --data_path data/tr_data/train_mathdial.json \
  --output_dir finetuned_models/trt_qwen_2point5_7b \
  --cache_dir data/cache \
  --model_max_length 2048 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 5 \
  --deepspeed deepspeed/config.json
```

**Checkpointing**: Use `--resume_from_checkpoint <path>` to resume from a saved checkpoint.

---

## Inference

Generate tutor responses with `inference/talker_predict.py`:

```bash
python inference/talker_predict.py \
  --adapter_path finetuned_models/trt_qwen_2point5_7b \
  --test_json data/tr_data/test_mathdial.json \
  --output_json data/tr_data/predictions.json
```

This will load the saved model checkpoint and generate next best teacher dialogue for the given conversation history.

---

## Prompts

* **`talker_prompt.txt`**: details the Socratic instruction template used by the talker.
* **`reasoner_prompt.txt`**: defines how to generate chain of thought and belief states from given info.
