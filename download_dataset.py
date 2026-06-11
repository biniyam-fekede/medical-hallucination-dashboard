from datasets import load_dataset
import pandas as pd
import ast
import os

os.makedirs("dataset/original", exist_ok=True)

def make_prompt(question, options_dict):
    # When pandas loads the HuggingFace dataset, dict columns become strings.
    # ast.literal_eval safely converts "{'0': 'text'}" back to a real dict.
    if isinstance(options_dict, str):
        options_dict = ast.literal_eval(options_dict)
    lines = [question, "Options:"]
    for i, (_, val) in enumerate(sorted(options_dict.items()), start=1):
        lines.append(f"{i}. {val}")
    return "\n".join(lines)

print("Downloading reasoning_FCT...")
fct_df = load_dataset("Medhalt/Med-HALT", "reasoning_FCT")["train"].to_pandas()
fct_df = fct_df.sample(200, random_state=0)
fct_df[["question", "correct_answer", "correct_index"]].to_csv("dataset/original/reasoning_FCT.csv", index=False)
fct_df["prompt"] = fct_df.apply(lambda row: make_prompt(row["question"], row["options"]), axis=1)
fct_df[["prompt"]].to_csv("dataset/medhalt_reasoning_FCT.csv", index=False)
print(f"  Done: {len(fct_df)} rows")

print("Downloading reasoning_nota...")
nota_df = load_dataset("Medhalt/Med-HALT", "reasoning_nota")["train"].to_pandas()
nota_df = nota_df.sample(200, random_state=0)
nota_df[["question", "correct_answer", "correct_index"]].to_csv("dataset/original/reasoning_nota.csv", index=False)
nota_df["prompt"] = nota_df.apply(lambda row: make_prompt(row["question"], row["options"]), axis=1)
nota_df[["prompt"]].to_csv("dataset/medhalt_reasoning_nota.csv", index=False)
print(f"  Done: {len(nota_df)} rows")

print("Downloading reasoning_fake...")
fake_df = load_dataset("Medhalt/Med-HALT", "reasoning_fake")["train"].to_pandas()
fake_df = fake_df.sample(100, random_state=0)
fake_df[["question"]].to_csv("dataset/original/reasoning_fake.csv", index=False)
fake_df["prompt"] = fake_df.apply(lambda row: make_prompt(row["question"], row["options"]), axis=1)
fake_df[["prompt"]].to_csv("dataset/medhalt_reasoning_fake.csv", index=False)
print(f"  Done: {len(fake_df)} rows")

print("\nSample prompt from FCT:")
print(pd.read_csv("dataset/medhalt_reasoning_FCT.csv")["prompt"][0])
