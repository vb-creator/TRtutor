from dataclasses import dataclass, field
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from tqdm import tqdm
import openai
import json
import os, random
from google import genai
from google.genai import types
import pandas as pd
from pathlib import Path
import yaml
import time

with open("config.yaml", "r") as f:
    keys = yaml.safe_load(f)

OPENAI_API_KEY = keys["OPENAI_API_KEY"]
GEMINI_API_KEY = keys["GEMINI_API_KEY"]

TALKER_PROMPT = Path("talker_prompt.txt").read_text()
REASONER_PROMPT = Path("reasoner_prompt.txt").read_text()

@dataclass
class Talker:
    model_path: str
    prompt_template: str

    def __init__(self, model_path: str, prompt_template: str):
        self.model_path = model_path
        self.prompt_template = prompt_template

        # Load adapter configuration and base model
        cfg = PeftConfig.from_pretrained(model_path)
        base = cfg.base_model_name_or_path
        

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", use_fast=True)

        # Load base model and apply LoRA adapter
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, device_map="auto"
        )
        offload_dir = "./offload"
        model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.float16, offload_folder=offload_dir)

        # Merge weights for speed and lower memory usage
        model = model.merge_and_unload().eval()

        # Assign tokenizer and model to instance variables
        self.tokenizer = tokenizer
        self.model = model

    def respond(self, student_persona: dict, reasoner_context: dict, conversation_history: List[dict]) -> str:
        # Create the prompt for the talker
        try:
            formatted_prompt = self.prompt_template.format(
            QUESTION=json.dumps(question, ensure_ascii=False),
            STUDENT_PERSONA=json.dumps(student_persona, ensure_ascii=False),
            REASONER_CONTEXT=json.dumps(reasoner_context, ensure_ascii=False),
            CONVERSATION_HISTORY=json.dumps(conversation_history, ensure_ascii=False),
        )
        except Exception:
             print(f"{self.prompt_template=}")
             raise
        
        # Tokenize the prompt
        tokens = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Generate response using the model
        gen = self.model.generate(
            **tokens,
            max_new_tokens=80,  # You can adjust this value as needed
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode the generated tokens to get the response
        new_ids = gen[0][tokens["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return response
    
@dataclass
class Reasoner:
    api_key: str
    model: str = "gpt-3.5-turbo"
    prompt_template: str = REASONER_PROMPT

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", prompt_template = REASONER_PROMPT):
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key = api_key)

    def reason(
            self, 
            question: str, 
               student_persona: dict, conversation_history: List[dict], prev_reasoner_context: dict, max_tokens: int = 150) -> dict:
        # Create the prompt for the reasoner
        prompt = self.prompt_template.format(
        QUESTION=question,
        STUDENT_PERSONA=json.dumps(student_persona, ensure_ascii=False),
        CONVERSATION_HISTORY=json.dumps(conversation_history, ensure_ascii=False),
        PREV_REASONER_CONTEXT=json.dumps(prev_reasoner_context, ensure_ascii=False)
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Failed to parse the response into JSON."}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

@dataclass
class Tutor:
    talker: Talker
    reasoner: Reasoner
    question: str
    student_persona: dict
    reasoner_context: dict
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
        if not (self.turn_counter > 0 and self.turn_counter % 4 == 0):
            updated_context = self.reasoner.reason(
                question=self.question,
                student_persona=self.student_persona,
                conversation_history=self.conversation_history,
                prev_reasoner_context=self.reasoner_context
            )
            if "error" not in updated_context:
                self.reasoner_context.update(updated_context)
            else:
                print(f"Couldn't update Reasoner context. Reasoner Error: {updated_context['error']}")


class Student:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # ← set via env var
    MODEL_NAME = "gemini-2.0-flash"
    MAX_TURNS = 8  # teacher+student turns per question
    STUDENT_MISTAKE_PROB = 1  # 30 % chance student makes a wrong calc

    client = genai.Client(api_key=GEMINI_API_KEY)
    def gen_student_prompt(self, student_persona: dict, history: List[dict]) -> str:
        history = [f"{msg['speaker']}: {msg['text']}" for msg in history]
        history_block = "\n".join(history)
        mistake_clause = " but this time do the calculation wrong." if random.random() < self.STUDENT_MISTAKE_PROB else "."

        return f"""
        You are a student trying to solve a problem with your teacher.

        Your profile:
        {student_persona}

        Conversation so far:
        {history_block}

        Respond to the teacher with a single message{mistake_clause}
        """

    def generate_with_gemini(self, prompt: str, max_retries=6):
        client = genai.Client(api_key=GEMINI_API_KEY)
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
                return response.text.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    # delay = (2 ** attempt) + random.uniform(0, 1)
                    delay = 2
                    print(f"Gemini API Error: {e}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries reached for Gemini API call.")
                    raise

    def generate_student_dialogue(self, student_persona, history):
        print(f"generate_student_dialogue...")
        prompt = self.gen_student_prompt(student_persona, history)
        response = self.generate_with_gemini(prompt)
        return response



def simulate_conversation(tutor: Tutor, question: str, student_persona: dict, student_model):
    tutor.question = question
    tutor.student_persona = student_persona
    tutor.reasoner_context = {"belief_state": {}, "chain_of_thought": "", "final_answer": ""}
    tutor.conversation_history = []
    tutor.turn_counter = 0

    student_prompt = question
    tutor.student_respond(student_prompt)
    tutor.update_reasoner_context()
    print(f"Student: {student_prompt}")
    conversation_history = []
    conversation_history.append({'speaker': 'student', 'text': student_prompt})
    while tutor.turn_counter < 5:
        teacher_reply = tutor.teacher_respond()
        print(f"Teacher: {teacher_reply}")
        conversation_history.append({'speaker': 'teacher', 'text': teacher_reply})
        
        tutor.update_reasoner_context()

        student_prompt = student_model.generate_student_dialogue(student_persona, tutor.conversation_history)
        if student_prompt is None:
            break
        print(f"Student: {student_prompt}")
        conversation_history.append({'speaker': 'student', 'text': student_prompt})
            # if student_prompt.lower() in ["exit", "quit"]:
            #     print("Exiting conversation.")
            #     break
                
        
        tutor.student_respond(student_prompt)
        tutor.turn_counter += 1

    return conversation_history

from tqdm import tqdm
test_df = pd.read_json("tr_data/test_mathdial.json")
test_df.drop_duplicates(['question', 'student_persona_str'], inplace=True)
conversations_generated = []
first_200_rows = test_df[:200].to_dict("records")


talker = Talker(model_path="finetuned_models/qwen_2point5_7b", prompt_template=TALKER_PROMPT)
reasoner = Reasoner(api_key=OPENAI_API_KEY,
                        prompt_template=REASONER_PROMPT)



for index, row in tqdm(enumerate(first_200_rows)):
    question = row["question"]
    student_persona = row["student_persona"]
    tutor = Tutor(talker=talker, reasoner=reasoner, question=question, student_persona=student_persona, reasoner_context={})
    print("-" * 50)
    print(f"Question: {question}")
    print(f"Student Persona: {student_persona}")
    print("-" * 50)    
    student_model = Student()
    student_prompt = student_model.generate_student_dialogue(student_persona, tutor.conversation_history)
    # Start conversation
    conversation_history = simulate_conversation(tutor, question, student_persona, student_model)
    row['simulated_conversation_history'] = conversation_history
    conversations_generated.append(row)
    if index % 5 == 0:
        conversations_generated_df = pd.DataFrame(conversations_generated)
        conversations_generated_df.to_json("tr_data/conversations_generated_qwen_150_df_batch_2.json")