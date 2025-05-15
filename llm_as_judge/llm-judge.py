import os
from openai import OpenAI 
from tqdm import tqdm
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SYSTEM_RUBRIC = """
You are an expert evaluator. Grade a math tutor’s reply using five criteria:
Coherence, Relevance, Personalization, Engagement, and Instructional Quality.
Each criterion is scored from 0 to 5 (0 = very poor, 5 = excellent).

Return your response as a valid JSON object in the exact format below:

{
  "Coherence": {
    "score": <0-5>,
    "justification": "<one-sentence reason for the Coherence score>"
  },
  "Relevance": {
    "score": <0-5>,
    "justification": "<one-sentence reason for the Relevance score>"
  },
  "Personalization": {
    "score": <0-5>,
    "justification": "<one-sentence reason for the Personalization score>"
  },
  "Engagement": {
    "score": <0-5>,
    "justification": "<one-sentence reason for the Engagement score>"
  },
  "Instructional_Quality": {
    "score": <0-5>,
    "justification": "<one-sentence reason for the Instructional Quality score>"
  }
}
""".strip()

LONG_RUBRIC_EXAMPLES = """
### Detailed criteria & examples (reference)

1 · Coherence  
5 – Clear, step‑by‑step: “Flip 1/3 to 3/1, multiply: 1/2×3/1 = 3/2.”  
3 – Mostly clear but with small gaps.  
0 – Disorganized/confusing.

2 · Relevance  
5 – Directly answers the student’s question.  
3 – Partially relevant.  
0 – Off‑topic.  
Example 0: Student asks about fractions; tutor talks about primes.

3 · Personalization  
5 – Tailors examples to the student’s interests or prior turns.  
3 – Neutral.  
0 – Generic or tone‑deaf.

4 · Engagement  
5 – Encouraging, motivating.  
3 – Factual but flat.  
0 – Dismissive (“Figure it out yourself.”).

5 · Instructional Quality  
5 – Correct reasoning and helpful examples.  
3 – Some help but unclear or shallow.  
0 – Wrong or misleading (“Add numerators and denominators …”).
""".strip()


# LONG_RUBRIC_EXAMPLES = """
# ### Scoring criteria & examples

# 1. **Coherence (0–5)**  
#    - Measures how logically structured and easy to follow the tutor's response is.  
#    - 5: The explanation is clear, step-by-step, and free from contradictions.  
#    - 3: Mostly makes sense, but may contain minor gaps or jumps.  
#    - 0: The response is disorganized or confusing.

#    **Examples:**  
#    - (5) “To divide, flip the second fraction and multiply: 1/2 ÷ 1/3 = 1/2 × 3/1 = 3/2.”  
#    - (3) “Divide by flipping it. You’ll get 3/2.”  
#    - (0) “You have to subtract the numerators somehow…”

# 2. **Relevance (0–5)**  
#    - Evaluates whether the tutor's reply directly addresses the student's question or concern.  
#    - 5: Fully focused and responsive.  
#    - 3: Partially related or contains some misunderstanding.  
#    - 0: Off-topic or ignores the student's intent.

#    **Examples:**  
#    - (5) Student: “How do I multiply fractions?” → Tutor explains multiplication clearly.  
#    - (0) Tutor responds with: “Let's learn about prime numbers.”

# 3. **Personalization (0–5)**  
#    - Assesses whether the response is adapted to the student’s persona, learning style, or previous context.  
#    - 5: Tailored examples or tone that aligns with interests, age, or prior steps.  
#    - 3: Neutral response with no clear personalization.  
#    - 0: Generic, robotic, or mismatched tone.

#    **Examples:**  
#    - (5) “Since you love baseball, imagine dividing a pizza at the game…”  
#    - (0) “Let me recite the multiplication steps without any context.”

# 4. **Engagement (0–5)**  
#    - Reflects how well the tutor keeps the student interested and motivated.  
#    - 5: Encouraging, positive, and invites curiosity.  
#    - 3: Factual but emotionally flat.  
#    - 0: Dismissive, boring, or discouraging.

#    **Examples:**  
#    - (5) “Awesome question! You’re thinking like a mathematician!”  
#    - (3) “Here’s the method.”  
#    - (0) “Figure it out yourself.”

# 5. **Instructional Quality (0–5)**  
#    - Indicates how effectively the tutor helps the student understand the concept.  
#    - 5: Clear teaching, examples, correct reasoning.  
#    - 3: Somewhat helpful but may lack depth or clarity.  
#    - 0: Incorrect, misleading, or no explanation.

#    **Examples:**  
#    - (5) “To add 1/2 and 1/4, use a common denominator: 2/4 + 1/4 = 3/4.”  
#    - (0) “Add both tops and bottoms: 1/2 + 1/4 = 2/6.”

# """.strip()

def judge_conversation(question: str, student_persona: str, conv: str, ground_truth: str) -> str:
    """
    conv: the raw dialogue text (Student/Teacher lines).
    returns: GPT‑4o's grading string.
    """
    user_prompt = f"\n-- Question Start --\n{question}\n-- Question End --\n-- Student Persona Start --\n{student_persona}\n-- Student Persona End --\n-- Conversation Start --\n{conv}\n-- Conversation End --\n-- Ground Truth Start --\n{ground_truth}\n-- Ground Truth End --\n"

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_RUBRIC},
            {"role": "system", "content": LONG_RUBRIC_EXAMPLES},
            {"role": "user", "content": user_prompt}
        ],
    )
    return response.choices[0].message.content.strip()



sample_prompts = [
    [
        "Nancy is filling an aquarium for her fish. She fills it halfway and goes to answer the door. While she's gone, her cat knocks the aquarium over and spills half the water in it. Then Nancy comes back and triples the amount of water in the aquarium. If the aquarium is 4 feet long, 6 feet wide, and 3 feet high, how many cubic feet of water are in the aquarium?",
        "I would have 12 magical artifacts.",
        "She would have 54 cubic feet of water.",
        "Yes, that makes sense. I just got confused because I thought I needed to add another step after finding the answer.",
        "Got it! Thanks for explaining it with the artifacts. That helps me see it more clearly."
    ],
    [
        "Nancy is filling an aquarium for her fish. She fills it halfway and goes to answer the door. While she's gone, her cat knocks the aquarium over and spills half the water in it. Then Nancy comes back and triples the amount of water in the aquarium. If the aquarium is 4 feet long, 6 feet wide, and 3 feet high, how many cubic feet of water are in the aquarium?",
        "I would have 12 magical artifacts.",
        "She would have 54 cubic feet of water.",
        "Yes, that makes sense. I just got confused because I thought I needed to add another step after finding the answer.",
        "Got it! Thanks for explaining it with the artifacts. That helps me see it more clearly."
    ],
]

sample_ref_responses = [
    [
        "Steven, let's say you have a collection of 4 magical artifacts from your favorite fantasy novel. If you triple that collection, how many artifacts would you have?",
        "Great! Now, imagine Nancy has 18 cubic feet of water in her enchanted cauldron. If she triples that amount, how much water would she have?",
        "Exactly correct! Now, remember, when you triple a number, you multiply it by 3. It's like adding the same amount two more times. Does that make sense with your collection of artifacts?",
        "I understand. It's easy to overthink sometimes, especially when you're used to detailed steps. Just remember, once you multiply by 3, you're done with the tripling step.",
        "You're welcome! I'm glad the artifact example helped clarify things. If you have any more questions or need further examples, just let me know!"
    ],
    [
        "Steven, let's say you have a collection of 4 magical artifacts from your favorite fantasy novel. If you triple that collection, how many artifacts would you have?",
        "Great! Now, imagine Nancy has 18 cubic feet of water in her enchanted cauldron. If she triples that amount, how much water would she have?",
        "Exactly correct! Now, remember, when you triple a number, you multiply it by 3. It's like adding the same amount two more times. Does that make sense with your collection of artifacts?",
        "I understand. It's easy to overthink sometimes, especially when you're used to detailed steps. Just remember, once you multiply by 3, you're done with the tripling step.",
        "You're welcome! I'm glad the artifact example helped clarify things. If you have any more questions or need further examples, just let me know!"
    ],
]

llm_scores = []
# prompts = sample_prompts
# predicted_responses = sample_ref_responses
prompts = []
predicted_responses = []
# load conversations_generated_qwen_150_df.json and print rows and print first entry
import json
import pandas as pd
with open("conversations_generated_qwen_150_df.json", "r") as f:
    data = json.load(f)
    # print(data.keys())  # print keys of the JSON object

# questions = data["questions"]
questions = data["question"]
conversations = data["simulated_conversation_history"]
ground_truths = data["ground_truth"]
student_personas = data["student_persona"]

for question, student_persona, conv, ground_truth in tqdm(list(zip(questions.values(), student_personas.values(), conversations.values(), ground_truths.values()))[5:]):
    conv_str = ""
    for dialogue in conv:
        # print(dialogue)
        conv_str += f"{dialogue['speaker']} : {dialogue['text']}\n"

    # print(f"\n\nConversation:\n---\n{conv_str}\n---")



    # judge_output = ""
    judge_output = judge_conversation(question, student_persona, conv_str, ground_truth)
    judge_output = json.loads(judge_output)
    # print(f"LLM Output:\n---\n{judge_output}\n---")
    # save the question, student_persona, conv, ground_truth and judge_output to a jsonl file
    with open("llm_eval_results.jsonl", "a") as f:
        f.write(json.dumps({
            "question": question,
            "student_persona": student_persona,
            "conversation": conv,
            "ground_truth": ground_truth,
            "coherence": judge_output["Coherence"],
            "relevance": judge_output["Relevance"],
            "personalization": judge_output["Personalization"],
            "engagement": judge_output["Engagement"],
            "instructional_quality": judge_output["Instructional_Quality"],
        }) + "\n")

    # print(f"LLM Output:\n---\n{judge_output}\n---")
    
    

# print("LLM Scores:", llm_scores)