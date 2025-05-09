# python tutoring_session.py
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from openai import OpenAI
import yaml
import requests
import random
import logging
import datetime
import json
import re

import pandas as pd
df = pd.read_pickle("data/mathdial_df.pkl")

logging.basicConfig(
    filename="logs/reasoner.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Fireworks API key
with open("config.yaml", "r") as f:
    keys = yaml.safe_load(f)

reasoner_context_db = "tr_data/reasoner_context_tracker.json"

FIREWORKS_API_KEY = keys["FIREWORKS_API_KEY"]
client = OpenAI(api_key=FIREWORKS_API_KEY, base_url="https://api.fireworks.ai/inference/v1")

URL = "https://api.fireworks.ai/inference/v1/chat/completions"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {FIREWORKS_API_KEY}"
}
# Load prompt templates
TALKER_PROMPT = Path("talker_prompt.txt").read_text()
REASONER_PROMPT = Path("reasoner_prompt.txt").read_text()

DEEPSEEK_MODEL = "deepseek-r1-basic"
# LLAMA_MODEL = "llama-v2-7b-chat"
LLAMA_MODEL = "llama4-scout-instruct-basic"
MISTRAL_MODEL = "mistral-7b-instruct-v3"

# default personas
test_personas = [
    {
        "name": "Aiden",
        "gender": "male",
        "interests": "soccer, video games, friendly competitions",
        "personality": "outgoing, competitive, hands-on learner",
        "background": "loves turning any activity into a game; enjoys team environments and learns best through trial and error."
    },
    {
        "name": "Lila",
        "gender": "female",
        "interests": "drawing, music, nature walks",
        "personality": "quiet, creative, observant",
        "background": "expresses herself better through art and visuals than words; enjoys exploring ideas through creative storytelling."
    },
    {
        "name": "Jordan",
        "gender": "non-binary",
        "interests": "programming, puzzles, logic games",
        "personality": "curious, precise, enjoys deep thinking",
        "background": "often gets absorbed in long-term projects; loves asking 'why' and breaking problems into patterns and rules."
    },
    {
        "name": "Sofia",
        "gender": "female",
        "interests": "debate, current events, social media",
        "personality": "confident, verbal, quick thinker",
        "background": "thrives in discussion-based environments; likes to explain her reasoning and test ideas aloud."
    },
    {
        "name": "Marcus",
        "gender": "male",
        "interests": "skateboarding, fixing bikes, DIY videos",
        "personality": "active, mechanical-minded, improvisational",
        "background": "prefers learning by doing and experimenting; gets easily bored by lectures but loves building and solving real-world problems."
    },
    {
        "name": "Priya",
        "gender": "female",
        "interests": "reading fantasy novels, journaling, mythology",
        "personality": "imaginative, reflective, organized",
        "background": "absorbs information deeply; likes step-by-step thinking and often connects abstract concepts to stories or metaphors."
    },
    {
        "name": "Riya",
        "gender": "female",
        "interests": "art, puzzles, science fiction",
        "personality": "curious, imaginative, likes visual learning",
        "background": "gets excited by creative problem solving and visual metaphors"
    }
]


# to set a random question to probe
def set_question(filepath: str = None):
    idx = random.randint(0, len(df) - 1)
    question = df.iloc[idx]['question']
    return question

# to set a random user persona for testing
def set_user_persona():
    rand_index = random.randint(0, 5)
    return test_personas[rand_index]


def generate_with_deepseek(prompt):
    payload = {
        "model": f"accounts/fireworks/models/{DEEPSEEK_MODEL}",
        "max_tokens": 1000,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content.strip())
    except Exception:
        return content

def generate_with_mistral(prompt: str):
    payload = {
        "model": f"accounts/fireworks/models/{MISTRAL_MODEL}",
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content.strip())
    except Exception:
        return content

def generate_with_llama(prompt: str):
    payload = {
        "model": f"accounts/fireworks/models/{LLAMA_MODEL}",
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content.strip())
    except Exception:
        return content

@dataclass
class Talker:
    model_name: str  # e.g., "accounts/fireworks/models/qwen-2.5-7b-chat"
    prompt_template: str

    def respond(self, student_persona: dict, reasoner_context: dict, conversation_history: List[dict]) -> str:
        prompt = self.prompt_template.format(
                STUDENT_PERSONA=json.dumps(student_persona, ensure_ascii=False),
                REASONER_CONTEXT=json.dumps(reasoner_context, ensure_ascii=False),
                CONVERSATION_HISTORY=json.dumps(conversation_history, ensure_ascii=False)
            )

        response = generate_with_llama(prompt)
        return response


def parse_deepseek(raw: str):
    """
    DeepSeek often starts with <think>...</think> then prints the JSON.
    """
    try:
        # a) remove the <think> block
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)

        # b) take the first {...}
        snippet = re.search(r"\{.*\}", cleaned, flags=re.S).group(0)
        return json.loads(snippet)
    except Exception:
        return raw


@dataclass
class Reasoner:
    model_name: str  # e.g., "accounts/fireworks/models/deepseek-llm-7b-chat"
    prompt_template: str

    def update_context(self, question: str, student_persona: dict, conversation_history: List[dict],
                   prev_reasoner_context: dict):
        prompt = self.prompt_template.format(
        QUESTION=question,
        STUDENT_PERSONA=json.dumps(student_persona, ensure_ascii=False),
        CONVERSATION_HISTORY=json.dumps(conversation_history, ensure_ascii=False),
        PREV_REASONER_CONTEXT=json.dumps(prev_reasoner_context, ensure_ascii=False)
        )
        response = generate_with_deepseek(prompt)

    
        try:
            # parsed = json.loads(response)
            parsed = parse_deepseek(response)
            return {
            "belief_state": parsed.get("belief_state", prev_reasoner_context["belief_state"]),
            "chain_of_thought": parsed.get("chain_of_thought", prev_reasoner_context["chain_of_thought"]),
            "final_answer": parsed.get("final_answer", prev_reasoner_context["final_answer"])
            }, parsed.get("update", False)
        except Exception:
            print("Warning: Reasoner response can't be parsed. Keeping previous context.")
            # print(f"unable to parse: {response=}")
            logging.warning(f"Unable to parse reasoner output: {response=}")
            return prev_reasoner_context, False


@dataclass
class DialogueSession:
    talker: Talker
    reasoner: Reasoner
    student_persona: dict
    reasoner_context: dict
    question: str
    conversation_history: List[dict] = field(default_factory=list)
    turn_counter: int = 0

    def student_respond(self, msg: str):
        self.conversation_history.append({"speaker": "student", "text": msg})
        self.turn_counter += 1

    def teacher_respond(self) -> str:
        reply = self.talker.respond(self.student_persona, self.reasoner_context, self.conversation_history)
        self.conversation_history.append({"speaker": "teacher", "text": reply})
        self.turn_counter += 1
        return reply

    def update_reasoner_context(self):
        # update context with every 4 exchanges - could switch to 2?
        if self.turn_counter > 0 and self.turn_counter % 4 == 0:
            self.reasoner_context, is_updated = self.reasoner.update_context(
                question=self.question,
                student_persona=self.student_persona,
                conversation_history=self.conversation_history,
                prev_reasoner_context=self.reasoner_context
            )

            if is_updated:
                print(f"reasoner context is updated...")
                log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "student_persona": self.student_persona,
                "updated_reasoner_context": self.reasoner_context,
                "conversation_history": self.conversation_history
            }

                
                if os.path.exists(reasoner_context_db):
                    with open(reasoner_context_db, "r+", encoding="utf-8") as f:
                        data = json.load(f)
                        data.append(log_entry)
                        f.seek(0)
                        json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    with open(reasoner_context_db, "w", encoding="utf-8") as f:
                        json.dump([log_entry], f, indent=2, ensure_ascii=False)


# Example Usage
if __name__ == "__main__":
    # Define static persona and initial reasoner context
    persona =  set_user_persona()

    initial_context = {
        "chain_of_thought": "",
        "final_answer": "",
        "belief_state": "Student has not begun solving; awaiting first response."
    }

    # question = "Liam has 4 shelves with 9 books each. He buys 2 more shelves and fills each with 9 books. How many books does he have now?"
    # set a random question from mathdial each time
    question = set_question()
    
    print("Probing the teacher with the question that student needs help with...")
    print(f"{question=}")
    print(f"Current student's persona: \n {persona=}")

    print(f"{initial_context=}")
    
    # Initialize session
    # accounts/fireworks/models/qwen2p5-72b-instruct
    # accounts/fireworks/models/deepseek-r1-basic
    talker = Talker(model_name="accounts/fireworks/models/qwen-2.5-7b-chat", prompt_template=TALKER_PROMPT)
    reasoner = Reasoner(model_name="accounts/fireworks/models/deepseek-llm-7b-chat", prompt_template=REASONER_PROMPT)
    session = DialogueSession(talker, reasoner, persona, initial_context, question=question)
    print(f"Student probes the teacher with math question they need help with...")
    student_probe = question
    # First student turn
    session.student_respond(student_probe)
    print("Teacher:", session.teacher_respond())
    
    while True:
        # from second turn
        student_probe = input("Student (type 'exit' to end conversation):")
        
        if student_probe.lower() == 'exit':
            break
        else:
            session.student_respond(student_probe)
            print("Teacher:", session.teacher_respond())
            session.update_reasoner_context()
