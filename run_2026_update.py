"""
2026 Update: Run GPT-5.5 and Gemma-3-27B on Med-HALT via OpenRouter.
Only base, system_prompt, and CoT methods (no MedRAG / internet search).

Usage:
    python run_2026_update.py
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = None  # loaded from .env below

MODELS = {
    "gpt-5.5": "openai/gpt-5.5",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
}

DATASET_DIR = "./dataset"
OUTPUT_DIR = "./results"
MAX_SAMPLES = 50  # Match the paper's sample size
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

def load_env():
    global OPENROUTER_API_KEY
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    OPENROUTER_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env")


def make_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def call_llm(client: OpenAI, model_id: str, messages: list, max_retries: int = 3) -> str | None:
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
                time.sleep(1)
    return None


def run_base(client, model_id, prompt):
    return call_llm(client, model_id, [{"role": "user", "content": prompt}])


def run_prompting(client, model_id, prompt):
    return call_llm(client, model_id, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])


def run_cot(client, model_id, prompt):
    return call_llm(client, model_id, [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n{COT_INSTRUCTION}"},
        {"role": "user", "content": prompt},
    ])


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
    load_env()
    client = make_client()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  ({model_id})")
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

            data = pd.read_csv(task_path)
            if "prompt" not in data.columns:
                print(f"  No 'prompt' column in {task_file}")
                continue

            # Use only first MAX_SAMPLES to match paper
            data = data.head(MAX_SAMPLES)
            print(f"\n  Task: {task_file}  ({len(data)} samples)")

            results = []
            for idx, prompt in enumerate(tqdm(data["prompt"], desc=f"  {model_name}/{task_file}")):
                parsed = convert_to_format(prompt)

                base = run_base(client, model_id, prompt)
                prom = run_prompting(client, model_id, prompt)
                cot = run_cot(client, model_id, prompt)

                results.append({
                    "question": parsed["question"],
                    "options": parsed["options"],
                    "base_output": base,
                    "prompting_output": prom,
                    "cot_output": cot,
                    "medrag_output": None,
                    "internetsearch_output": None,
                })

                # Print progress every 10
                if (idx + 1) % 10 == 0:
                    print(f"    [{idx+1}/{len(data)}] last base: {repr((base or '')[:60])}")

                    # Save intermediate
                    inter_file = out_file.replace("_seed0.json", "_intermediate_seed0.json")
                    with open(inter_file, "w") as f:
                        json.dump({"seed": 0, "results": results}, f, indent=2)

                time.sleep(0.5)  # gentle rate limiting

            # Save final
            with open(out_file, "w") as f:
                json.dump({"seed": 0, "results": results}, f, indent=2)
            print(f"  Saved: {out_file}")

    print("\n All done! Run the dashboard to see updated results.")


if __name__ == "__main__":
    main()
