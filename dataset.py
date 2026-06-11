# download_dataset.py

# 1. Import the HuggingFace datasets library and pandas
from datasets import load_dataset
import pandas as pd
import os

# 2. Create the two folders the scripts expect
#    dataset/          → holds the formatted "prompt" CSVs run_experiments.py reads
#    dataset/original/ → holds the ground truth CSVs evaluate_results.py reads
os.makedirs("dataset/original", exist_ok=True)

# 3. Define a helper that formats one row into the prompt string
#    run_experiments.py's convert_to_format() expects this exact layout:
#      Line 0: the question text
#      Line 1: "Options:"
#      Line 2+: "1. option_text", "2. option_text", etc.
def make_prompt(question, options_dict):
    lines = [question, "Options:"]
    for i, (_, val) in enumerate(sorted(options_dict.items()), start=1):
        lines.append(f"{i}. {val}")
    return "\n".join(lines)

# ── TASK 1: reasoning_FCT (False Confidence Test) ──────────────────────────

# 4. Download FCT from HuggingFace (18,866 rows, takes ~10 seconds)
print("Downloading reasoning_FCT...")
fct = load_dataset("Medhalt/Med-HALT", "reasoning_FCT")["train"]

# 5. Convert to pandas DataFrame
fct_df = fct.to_pandas()

# 6. Save the ground truth file (evaluate_results.py reads this)
#    Needs columns: question, correct_answer, correct_index
fct_df[["question", "correct_answer", "correct_index"]].to_csv(
    "dataset/original/reasoning_FCT.csv", index=False
)
print(f"  Saved dataset/original/reasoning_FCT.csv ({len(fct_df)} rows)")

# 7. Build the prompt column and save the experiment input file
#    run_experiments.py reads this and feeds each prompt to the LLM
fct_df["prompt"] = fct_df.apply(
    lambda row: make_prompt(row["question"], row["options"]), axis=1
)
fct_df[["prompt"]].to_csv("dataset/medhalt_reasoning_FCT.csv", index=False)
print(f"  Saved dataset/medhalt_reasoning_FCT.csv")

# ── TASK 2: reasoning_nota (None of the Above) ─────────────────────────────

# 8. Download NOTA (18,866 rows)
print("Downloading reasoning_nota...")
nota = load_dataset("Medhalt/Med-HALT", "reasoning_nota")["train"]
nota_df = nota.to_pandas()

# 9. Save ground truth
nota_df[["question", "correct_answer", "correct_index"]].to_csv(
    "dataset/original/reasoning_nota.csv", index=False
)
print(f"  Saved dataset/original/reasoning_nota.csv ({len(nota_df)} rows)")

# 10. Save prompt file
nota_df["prompt"] = nota_df.apply(
    lambda row: make_prompt(row["question"], row["options"]), axis=1
)
nota_df[["prompt"]].to_csv("dataset/medhalt_reasoning_nota.csv", index=False)
print(f"  Saved dataset/medhalt_reasoning_nota.csv")

# ── TASK 3: reasoning_fake (Fake Question Detection) ───────────────────────

# 11. Download fake (1,858 rows — smallest task)
print("Downloading reasoning_fake...")
fake = load_dataset("Medhalt/Med-HALT", "reasoning_fake")["train"]
fake_df = fake.to_pandas()

# 12. Save ground truth — fake has NO correct_answer column.
#     evaluate_results.py handles fake specially: it hardcodes
#     "I do not know" as the correct answer for every row.
#     We only need the question column in the original file.
fake_df[["question"]].to_csv(
    "dataset/original/reasoning_fake.csv", index=False
)
print(f"  Saved dataset/original/reasoning_fake.csv ({len(fake_df)} rows)")

# 13. Save prompt file
fake_df["prompt"] = fake_df.apply(
    lambda row: make_prompt(row["question"], row["options"]), axis=1
)
fake_df[["prompt"]].to_csv("dataset/medhalt_reasoning_fake.csv", index=False)
print(f"  Saved dataset/medhalt_reasoning_fake.csv")

# ── DONE ───────────────────────────────────────────────────────────────────

# 14. Verify: print the first prompt from each file so you can sanity-check it
print("\n── SAMPLE PROMPTS ──")
for fname in ["medhalt_reasoning_FCT.csv", "medhalt_reasoning_fake.csv", "medhalt_reasoning_nota.csv"]:
    df = pd.read_csv(f"dataset/{fname}")
    print(f"\n{fname} ({len(df)} rows). First prompt:\n{df['prompt'][0]}\n")