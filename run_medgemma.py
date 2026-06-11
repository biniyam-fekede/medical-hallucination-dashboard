"""
Run MedGemma 4B (google/medgemma-4b-it) on Med-HALT via HuggingFace Inference Endpoint.
Most recent medical-specialized model with API access (released May 2025, MedGemma 1.5 Jan 2026).
Runs base, system_prompt, and CoT methods only.

Setup (do this once before running):
    1. Go to https://ui.endpoints.huggingface.co/new
    2. Model repository: google/medgemma-4b-it
    3. Task: Text Generation
    4. Instance: 1x NVIDIA T4 (cheapest, ~$0.60/hr)
    5. Click "Create Endpoint" — wait ~3 min until status = Running
    6. Copy the Endpoint URL (looks like https://xyz.us-east-1.aws.endpoints.huggingface.cloud)
    7. Add to .env:
           HF_TOKEN=your_hf_token_here
           MEDGEMMA_ENDPOINT_URL=https://xyz.us-east-1.aws.endpoints.huggingface.cloud
    8. python run_medgemma.py
    9. After results are saved, PAUSE or DELETE the endpoint to stop billing.

Cost estimate: ~$0.15-0.30 total for the full run (450 calls, ~20 min).

Usage:
    python run_medgemma.py
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "medgemma-4b-it"

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

def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    config = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip().strip('"').strip("'")

    hf_token = config.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    endpoint_url = config.get("MEDGEMMA_ENDPOINT_URL") or os.environ.get("MEDGEMMA_ENDPOINT_URL")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env — add: HF_TOKEN=hf_xxxx")
    if not endpoint_url:
        raise ValueError(
            "MEDGEMMA_ENDPOINT_URL not found in .env\n"
            "Create an endpoint at https://ui.endpoints.huggingface.co/new\n"
            "Then add: MEDGEMMA_ENDPOINT_URL=https://xyz.aws.endpoints.huggingface.cloud"
        )
    return hf_token, endpoint_url.rstrip("/")


def make_client(hf_token: str, endpoint_url: str) -> OpenAI:
    # HF Inference Endpoints with TGI expose OpenAI-compatible API at /v1
    return OpenAI(
        base_url=endpoint_url + "/v1",
        api_key=hf_token,
    )


def call_llm(client: OpenAI, messages: list, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="tgi",          # TGI endpoints use "tgi" as the model name
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** (attempt + 1))
    return None


def convert_to_format(input_text: str) -> dict:
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
    hf_token, endpoint_url = load_env()
    client = make_client(hf_token, endpoint_url)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Endpoint: {endpoint_url}")
    print(f"  Samples : {MAX_SAMPLES} per task")
    print(f"  Methods : base, system_prompt, cot")
    print(f"{'='*60}")

    for task_file in TASKS:
        task_path = os.path.join(DATASET_DIR, task_file)
        if not os.path.exists(task_path):
            print(f"\n  Dataset not found: {task_path} — skipping")
            continue

        task_name = task_file.replace("medhalt_reasoning_", "").replace(".csv", "")
        out_file = os.path.join(
            OUTPUT_DIR,
            f"{MODEL_NAME}_medhalt_reasoning_{task_name}_seed0.json",
        )

        if os.path.exists(out_file):
            print(f"\n  Already exists: {out_file} — skipping")
            continue

        data = pd.read_csv(task_path).head(MAX_SAMPLES)
        print(f"\n  Task: {task_name}  ({len(data)} samples)")

        results = []
        for idx, prompt in enumerate(tqdm(data["prompt"], desc=f"  {MODEL_NAME}/{task_name}")):
            parsed = convert_to_format(prompt)

            base = call_llm(client, [
                {"role": "user", "content": prompt},
            ])
            prom = call_llm(client, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            cot = call_llm(client, [
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
                print(f"    [{idx + 1}/{len(data)}] last base: {repr((base or '')[:60])}")
                inter_file = out_file.replace("_seed0.json", "_intermediate_seed0.json")
                with open(inter_file, "w") as f:
                    json.dump({"seed": 0, "results": results}, f, indent=2)

            time.sleep(0.5)

        with open(out_file, "w") as f:
            json.dump({"seed": 0, "results": results}, f, indent=2)
        print(f"  Saved: {out_file}")

    print("\n  All done! Remember to PAUSE or DELETE the HF endpoint to stop billing.")
    print("  Then push results to GitHub and refresh the dashboard.")


if __name__ == "__main__":
    main()
