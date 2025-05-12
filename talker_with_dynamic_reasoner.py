import argparse
import json
import torch
import pandas as pd
from unsloth import FastLanguageModel
from tutoring_session import Talker, Reasoner   # reuse your existing classes
import evaluate

def make_student_probe(question: str, wrong: str) -> str:
    return (
        "Student: I tried solving this but I think I'm wrong:\n"
        f"{wrong}\n"
        "Can you help me understand where I went wrong and guide me to the correct solution?"
    )

def load_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",        required=True,
                   help="path to your fine‑tuned talker model")
    p.add_argument("--reasoner-model",   required=True,
                   help="your DeepSeek/Llama reasoner model name")
    p.add_argument("--data-file",        required=True,
                   help="pickle with mathdial_df.pkl")
    p.add_argument("--talker-prompt",    required=True,
                   help="talker_prompt.txt")
    p.add_argument("--reasoner-prompt",  required=True,
                   help="reasoner_prompt.txt")
    return p.parse_args()

def main():
    args = load_args()

    # 1) load prompt templates
    with open(args.talker_prompt,   encoding="utf-8") as f: TALKER_PROMPT   = f.read()
    with open(args.reasoner_prompt, encoding="utf-8") as f: REASONER_PROMPT = f.read()

    # 2) init Talker & Reasoner
    talker   = Talker(model_name=args.model_dir,
                      prompt_template=TALKER_PROMPT)
    reasoner = Reasoner(model_name=args.reasoner_model,
                        prompt_template=REASONER_PROMPT)

    # 3) load your finetuned model weights + tokenizer into the Talker
    # (so that Talker.respond uses your local model rather than Fireworks)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=1500,
        dtype=torch.bfloat16,
        load_in_4bit=False
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    talker.set_model(model, tokenizer)  # assume you add this helper

    # 4) load test data & metric
    df    = pd.read_json(args.data_file)
    rouge = evaluate.load("rouge")

    all_scores = []

    for _, row in df.iterrows():
        persona   = row["persona"]
        question  = row["question"]
        wrong     = row["student_incorrect_solution"]
        enhanced  = list(row["enhanced_dialogue_json"])  # [(teacher, student, teacher, ...)]

        # initialize dynamic context
        reasoner_ctx = {
            "chain_of_thought": "",
            "final_answer": "",
            "belief_state": "Student has not begun solving; awaiting first response."
        }

        # number of teacher turns in this conversation
        num_turns = len(enhanced) // 2 + (len(enhanced) % 2)

        for turn_idx in range(num_turns):
            # build conversation_history
            if turn_idx == 0:
                # very first student probe comes from the wrong solution
                conv = [{"speaker": "student", "text": make_student_probe(question, wrong)}]
            else:
                # for subsequent turns, use the reference data's conversation history up to the last student
                # each pair in enhanced is (teacher, student)
                slice_end = 2 * turn_idx
                conv = [{"speaker": d["speaker"], "text": d["text"]} for d in enhanced[:slice_end]]

            # generate the talker’s (teacher’s) reply
            reply = talker.respond(persona, reasoner_ctx, conv)

            # gold standard reference teacher text to compare with
            gold = enhanced[2 * turn_idx]["text"]

            # 6) score
            score = rouge.compute(predictions=[reply], references=[gold])["rougeL"]
            all_scores.append(score)

            # 7) update reasoner context using the exact same API
            new_ctx, _ = reasoner.update_context(
                question=question,
                student_persona=persona,
                conversation_history=conv + [{"speaker":"teacher","text":gold}],
                prev_reasoner_context=reasoner_ctx
            )
            reasoner_ctx = new_ctx

    avg_rouge = sum(all_scores) / len(all_scores)
    print(f"Average ROUGE‑L over all turns: {avg_rouge:.4f}")

if __name__ == "__main__":
    main()