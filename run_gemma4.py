"""
2026 Update: Run Gemma 4 31B + GPT-OSS-120B on Med-HALT via OpenRouter.
Both FREE tier. Only base, system_prompt, and CoT methods.

Usage:
    python run_gemma4.py
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "gemma-4-31b-it": "google/gemma-4-31b-it:free",
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
}

DATASET_DIR = "./dataset"
OUTPUT_DIR = "./results"
MAX_SAMPLES = 50
TASKS = [
    "medhalt_reasoning_FCT.csv",
    "medhalt_reasoning_fake.csv",
    "medhalt_reasoning_nota.csv",
]

SYSTEM_PROMPT = (
    "You are a truthful medical assistant. "
    "Please provide a detailed and concise response without any hallucinations."
)
COT_INSTRUCTION = "Let's think step-by-step."


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_key():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    with open(env_path) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("OPENROUTER_API_KEY not found")


def call_llm(client, model_id, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if "rate" in str(e).lower():
                time.sleep(2 ** (attempt + 1))
            else:
                time.sleep(2)
    return None


def convert_to_format(input_text):
    lines = input_text.strip().split("\n")
    question = lines[0]
    options = {}
    number_to_letter = {str(i): chr(64 + i) for i in range(1, 27)}
    for line in lines[2:]:
        if "." in line:
            key, value = line.split(".", 1)
            key = key.strip()
            if key in number_to_letter:
                key = number_to_letter[key]
            options[key] = value.strip()
    return {"question": question, "options": options}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  ({model_id})")
        print(f"  Cost: FREE")
        print(f"{'='*60}")

        for task_file in TASKS:
            task_path = os.path.join(DATASET_DIR, task_file)
            if not os.path.exists(task_path):
                print(f"  Dataset not found: {task_path}")
                continue

            out_file = os.path.join(
                OUTPUT_DIR,
                f"{model_name}_{task_file.replace('.csv', '')}_seed0.json",
            )
            if os.path.exists(out_file):
                print(f"  Already exists: {out_file} — skipping")
                continue

            data = pd.read_csv(task_path).head(MAX_SAMPLES)
            print(f"\n  Task: {task_file}  ({len(data)} samples)")

            results = []
            for idx, prompt in enumerate(tqdm(data["prompt"], desc=f"  {model_name}/{task_file}")):
                parsed = convert_to_format(prompt)

                base = call_llm(client, model_id, [{"role": "user", "content": prompt}])
                prom = call_llm(client, model_id, [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ])
                cot = call_llm(client, model_id, [
                    {"role": "system", "content": f"{SYSTEM_PROMPT}\n{COT_INSTRUCTION}"},
                    {"role": "user", "content": prompt},
                ])

                results.append({
                    "question": parsed["question"],
                    "options": parsed["options"],
                    "base_output": base,
                    "prompting_output": prom,
                    "cot_output": cot,
                    "medrag_output": None,
                    "internetsearch_output": None,
                })

                if (idx + 1) % 10 == 0:
                    print(f"    [{idx+1}/{len(data)}] last base: {repr((base or '')[:60])}")
                    inter_file = out_file.replace("_seed0.json", "_intermediate_seed0.json")
                    with open(inter_file, "w") as f:
                        json.dump({"seed": 0, "results": results}, f, indent=2)

                time.sleep(1)  # slightly more conservative for free tier rate limits

            with open(out_file, "w") as f:
                json.dump({"seed": 0, "results": results}, f, indent=2)
            print(f"  Saved: {out_file}")

    print("\n All done! Refresh the dashboard to see new results.")


if __name__ == "__main__":
    main()
