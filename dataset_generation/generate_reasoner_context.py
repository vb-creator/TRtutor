"""
used to generate reasoner_context using deepseek-r1, 
o3-mini, llama-4-scount
"""
import requests
import yaml
import logging
import datetime
import re
import ast
import os
import json
import pandas as pd
from typing import List
from tqdm import tqdm
from pathlib import Path
# from unsloth import FastLanguageModel
# from transformers import AutoTokenizer
from openai import OpenAI
import swifter
from tqdm import tqdm
import pandas as pd
import ast

# Configure logging
logging.basicConfig(
    filename="logs/talker_generate.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load API key from config
with open("config.yaml", "r") as f:
    keys = yaml.safe_load(f)

FIREWORKS_API_KEY = keys.get("FIREWORKS_API_KEY")
client = OpenAI(api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1")

# File to track reasoner context updates
reasoner_context_db = "tr_data/reasoner_context_tracker.json"

# Load prompt templates
TALKER_PROMPT = Path("talker_prompt.txt").read_text()
REASONER_PROMPT = Path("reasoner_prompt.txt").read_text()

# Default personas for testing
with open("tr_data/default_personas.json", "r") as f:
    test_personas = json.load(f)

LOCALE = "https://api.fireworks.ai/inference/v1/chat/completions"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {FIREWORKS_API_KEY}"
}

def generate_with_deepseek(prompt: str):
    """
    Call DeepSeek API, return JSON or raw content.
    """
    payload = {
        "model": "accounts/fireworks/models/deepseek-r1-basic",
        "max_tokens": 1000,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(LOCALE, headers=HEADERS, json=payload)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content.strip())
    except Exception:
        return content


talker_template = Path('talker_prompt.txt').read_text()
reasoner_template = Path('reasoner_prompt.txt').read_text()

prompt_template = reasoner_template

def generate_reasoner_context(
        question: str,
        student_persona: dict,
        conversation_history: List[dict],
        prev_reasoner_context: dict
    ):
        """
        Use DeepSeek to parse and return updated reasoning context.
        """
        prompt = prompt_template.format(
            QUESTION=question,
            STUDENT_PERSONA=json.dumps(student_persona, ensure_ascii=False),
            CONVERSATION_HISTORY=json.dumps(conversation_history, ensure_ascii=False),
            PREV_REASONER_CONTEXT=json.dumps(prev_reasoner_context, ensure_ascii=False)
        )
        raw = generate_with_deepseek(prompt)
        # Extract JSON snippet
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        match = re.search(r"\{.*?\}", cleaned, flags=re.S)
        try:
            parsed = json.loads(match.group(0)) if match else {}
        except Exception:
            parsed = {}
        new_ctx = {
            "belief_state": parsed.get("belief_state", prev_reasoner_context.get("belief_state")),
            "chain_of_thought": parsed.get("chain_of_thought", prev_reasoner_context.get("chain_of_thought")),
            "final_answer": parsed.get("final_answer", prev_reasoner_context.get("final_answer"))
        }
        updated = bool(parsed.get("update", False))
        if updated:
            logging.info("Reasoner context updated.")
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "student_persona": student_persona,
                "updated_reasoner_context": new_ctx,
                "conversation_history": conversation_history
            }
            if os.path.exists(reasoner_context_db):
                with open(reasoner_context_db, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data.append(entry)
                    f.seek(0)
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(reasoner_context_db, "w", encoding="utf-8") as f:
                    json.dump([entry], f, indent=2, ensure_ascii=False)
        return new_ctx, updated


sft_df = pd.read_json("tr_data/mathdial_sft_extended.jsonl", lines=True)
sft_df['input_json'] = sft_df['input'].apply(lambda x: ast.literal_eval(x))
sft_data = sft_df.to_dict("records")

deepseek_contexts = []


for s in tqdm(sft_data):
    s['deepseek_reasoner_context']  = generate_reasoner_context(s['question'], s['input_json']['student_persona'], s['input_json']['conversation_history'], s['input_json']['reasoner_context'])
    deepseek_contexts.append(s)
    if len(deepseek_contexts)%50 == 0:
        pd.DataFrame(deepseek_contexts).to_json("tr_data/deepseek_reasoner_context.json")        