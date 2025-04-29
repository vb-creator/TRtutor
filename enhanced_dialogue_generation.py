import pandas as pd
import random
import re
import json
import os
from pathlib import Path
import openai
from typing import Dict, Any, List
import openai
from openai import OpenAI
import yaml
from tqdm import tqdm
import sys


with open("config.yaml", "r") as f:
    keys = yaml.safe_load(f)
    # client = OpenAI(api_key=keys["OPEN_API_KEY"])

client = OpenAI(api_key=keys["OPENAI_API_KEY"], organization=keys["OPENAI_ORG_KEY"])

# set model
METADATA_MODEL = "gpt-4o"     # model for metadata
OPENAI_REASON_MODEL = "o3-mini"    # model for Aryabhata reasoning

client_o3   = openai.OpenAI(api_key=keys["OPENAI_API_KEY"], organization=keys["OPENAI_ORG_KEY"])
client_gpt4 = openai.OpenAI(api_key=keys["OPENAI_API_KEY"], organization=keys["OPENAI_ORG_KEY"])


mathdial_df = pd.read_pickle("data/mathdial_df.pkl")
# determine starting record index from command-line argument
start_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0


mathdial_records = mathdial_df.to_dict("records")

print(f"{start_index=}, {len(mathdial_records)=}\n{mathdial_records[start_index]=}\n{len(mathdial_records[start_index:])=}")

# default personas
personas = [
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
    }
]


def extract_name_gender(profile_text):
    # Extract name (assumes first word before "is")
    name_match = re.search(r"(\w+)\s+is", profile_text)
    name = name_match.group(1) if name_match else None
    if name:
        gender = "female" if name.lower().endswith('a') else "male"
    else:
        gender = None
    return {"name": name, "gender": gender}

def set_persona(x):
    rand_index = random.randint(0, 5)
    x['interests'] = personas[rand_index]['interests']
    x['personality'] = personas[rand_index]['personality']
    x['background'] = personas[rand_index]['background']
    return x


"""
1.  Reasoner-context generation 
2.  Dialogue enhancement 

Mathdial example format:
{
  "id": ...,
  "question": ...,
  "answer": ...,
  "student_incorrect_solution": ...,
  "teacher_described_confusion": ...,
  "dialogue": [                       # list of {"speaker": "...", "text": "..."}
      {...}, {...}, ...
  ],
  "persona": {                       # exactly this structure
      "name": ...,
      "gender": ...,
      "interests": ...,
      "personality": ...,
      "background": ...,
  }
}
"""


def generate_reasoner_context(example):
    """
    Returns a dict with keys  chain_of_thought, final_answer, belief_state
    using the o3-mini model.
    """
    system = (
        "You are an expert math tutor. "
        "Given a problem, produce a JSON object with:\n"
        "  chain_of_thought – a clear step-by-step solution.\n"
        "  final_answer     – the numeric or textual answer only.\n"
        "  belief_state     – short summary of where the student is confused, "
        "                     based on their incorrect solution and the teacher note.\n"
        "Respond with *only* valid JSON."
    )
    user = {
        "question": example["question"],
        "correct_answer": example["ground_truth"],
        "student_incorrect_solution": example["student_incorrect_solution"],
        "teacher_described_confusion": example["teacher_described_confusion"],
        "persona_background": example["persona"]["background"],
    }
    resp = client.chat.completions.create(model=OPENAI_REASON_MODEL,
    messages=[
        {"role": "system", "content": system},
        {"role": "user",   "content": json.dumps(user, ensure_ascii=False)}
    ])
    try:
        return json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return resp.choices[0].message.content.strip()

    # resp = client_o3.chat.completions.create(
    #     model="o3-mini",
    #     messages=[
    #         {"role": "system", "content": sys},
    #         {"role": "user",   "content": json.dumps(user, ensure_ascii=False)}
    #     ],
    #     temperature=0.3,
    # )
    # return json.loads(resp.choices[0].message.content)


def enhance_dialogue(example):
    """
    Returns a *new* dialogue (list of turns) whose teacher utterances are
    personalised; student turns are lightly edited for coherence.
    Uses gpt-4o.
    """
    system = (
        "You are rewriting a teacher-student conversation.\n"
        "Requirements:\n"
        "You can include more dialogues if needed, to simulate the student's confusion, and the teacher's attempt to personalize the response to the student's background"
        "2. Update teacher turns so they:\n"
        "include any reference to student's interests/hobbies if applicable but minimally \n"
        "3. Keep student turns coherent\n"
        "try to reflect the student's confusion when they respond to teacher's explanation initially, that simulates the mistake the student originally did when they solve the problem, and have the teacher correct it through relevant examples"
        f"try to include a few dialogues from the student side that communicates their incorrect understanding or confusion, in this case the student was confused about {example['teacher_described_confusion']}"
        "4. Do not reveal chain-of-thought or belief_state to either party.\n"
        "Return *only* a JSON array of turns in the form "
        "[{\"speaker\": \"student\"|\"teacher\", \"text\": \"...\"}, ...]."
    )

    user_payload = {
        "persona": example["persona"],
        "reasoner_context": example['reasoner_context'],       # chain_of_thought + belief_state + final_answer
        "dialogue": example["conversation"]
    }

    resp = client_gpt4.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        temperature=0.4,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return resp.choices[0].message.content
    


for i, eg in enumerate(tqdm(mathdial_records[start_index:])):
    try:
        reasoner_context = generate_reasoner_context(eg)
    except Exception as e:
        print(f"{eg=}")
        # raise
        reasoner_context = {"error": str(e)}
    mathdial_records[start_index+i]['reasoner_context'] = str(reasoner_context)
    try:
        enhanced_dialogue = enhance_dialogue(eg, reasoner_context)
    except Exception as e:
        enhanced_dialogue = {"error": str(e)}
    mathdial_records[start_index+i]['enhanced_dialogue'] = str(enhanced_dialogue)

    # update results for every 5 records to data/mathdial_enhanced.json
    if (i + 1) % 5 == 0:
        with open('data/mathdial_enhanced.json', 'a') as f:
            for entry in mathdial_records[start_index+i-4:start_index+i+1]:
                f.write(json.dumps(entry) + '\n')