import pandas as pd
import json
from pathlib import Path

def prepare_sft_data(
    enhanced_pkl: str,
    talker_prompt_path: str,
    output_pickle: str,
    output_jsonl: str = None
):
    """
    Reads a pickle file with full conversations, flattens into SFT examples,
    and writes out a DataFrame with columns: instruction, input, output.
    
    - enhanced_pkl: path to mathdial_enhanced.pkl
    - talker_prompt_path: path to talker_prompt.txt (the static instruction)
    - output_pickle: where to save the resulting SFT DataFrame (.pkl)
    - output_jsonl: (optional) where to also save JSONL of examples
    """
    # Load static instruction
    instruction = Path(talker_prompt_path).read_text()

    # Load enhanced conversations
    df = pd.read_pickle(enhanced_pkl)

    sft_records = []
    for _, row in df.iterrows():
        persona = row["persona"]
        reasoner_ctx = row["reasoner_context_json"]
        convo = list(row["enhanced_dialogue_json"])

        history = []
        for turn in convo:
            if turn["speaker"] == "teacher":
                # Build the SFT example
                inp = {
                    "student_persona": persona,
                    "reasoner_context": reasoner_ctx,
                    "conversation_history": history
                }
                sft_records.append({
                    "instruction": instruction,
                    "input": json.dumps(inp, ensure_ascii=False),
                    "output": turn["text"]
                })
            # Add every turn to history for next example
            history.append(turn)

    # Save as pickle
    out_df = pd.DataFrame(sft_records)
    Path(output_pickle).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(output_pickle)
    print(f"Saved {len(sft_records)} SFT examples to {output_pickle}")

    # Optionally save as JSONL
    if output_jsonl:
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for rec in sft_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Also saved JSONL to {output_jsonl}")


# usage
"""
prepare_sft_data(
    enhanced_pkl="tr_data/enhanced_mathdial_conversations.pkl",
    talker_prompt_path="sft_talker_prompt.txt",
    output_pickle="tr_data/mathdial_sft.pkl",
    output_jsonl="tr_data/mathdial_sft.jsonl"
)
"""