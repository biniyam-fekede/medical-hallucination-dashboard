"""
Targeted MedRAG patch: fills in medrag_output=None entries in existing result JSONs.
Loads one model at a time, runs retrieval + generation only for missing samples.

Usage:
    python patch_medrag.py --models medalpaca-13b AlpaCare-llama2-13b \
        --results_dir /tmp/biniyam_results \
        --corpus_dir ./MedRAG/corpus

Run AFTER MedGemma finishes (only one 13B model on GPU at a time).
"""

import os
import sys
import json
import argparse
import gc
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, './MedRAG/src')

TASKS = ["FCT", "fake", "nota"]

MODEL_CLASS_MAP = {
    "medalpaca-13b": ("medalpaca", "medalpaca/medalpaca-13b"),
    "AlpaCare-llama2-13b": ("xz97", "xz97/AlpaCare-llama2-13b"),
}


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_path, cache_dir="./model_cache"):
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path} (4-bit quant) ...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_cfg,
        offload_folder=f"./offloaded_weights_{model_path.replace('/', '_')}",
    )
    model.eval()
    print(f"Model loaded.")
    return tokenizer, model


def gen_response(tokenizer, model, prompt, max_new_tokens=512):
    clear_gpu()
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    input_len = inputs.input_ids.shape[1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_rag_prompt(question, options, context):
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return (
        "Use the following medical references to answer the question.\n\n"
        "References:\n" + context + "\n\n"
        "Question: " + question + "\n\nOptions:\n" + opts + "\n\n"
        "Select the correct option letter."
    )


def patch_file(json_path, tokenizer, model, retrieval_system):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    patched = 0

    for i, sample in enumerate(results):
        if sample.get("medrag_output") is not None:
            continue  # already filled

        question = sample.get("question", "")
        options = sample.get("options", {})
        if not question:
            continue

        print(f"  Sample {i+1}/{len(results)}: running MedRAG ...")
        try:
            retrieved_snippets, scores = retrieval_system.retrieve(question, k=5)
            contexts = []
            for idx, snippet in enumerate(retrieved_snippets[:5]):
                title = snippet.get("title", "Unknown")
                body = snippet.get("content", "")
                contexts.append(f"[{idx+1}] {title}\n{body}")
            context = "\n\n".join(contexts)

            rag_prompt = build_rag_prompt(question, options, context)
            answer = gen_response(tokenizer, model, rag_prompt)
            sample["medrag_output"] = answer
            print(f"    -> {repr(answer[:80])}")
            patched += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            sample["medrag_output"] = None

    if patched > 0:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {patched} new MedRAG outputs to {json_path}")
    else:
        print(f"  Nothing to patch in {json_path}")

    return patched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--results_dir", default="/tmp/biniyam_results")
    parser.add_argument("--corpus_dir", default="./MedRAG/corpus")
    parser.add_argument("--cache_dir", default="./model_cache")
    args = parser.parse_args()

    # Load retrieval system once (shared across models)
    print("Loading MedCPT retrieval system ...")
    from utils import RetrievalSystem
    retrieval_system = RetrievalSystem(
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        db_dir=args.corpus_dir,
        cache=True,
    )
    print("Retrieval system ready.")

    for model_name in args.models:
        if model_name not in MODEL_CLASS_MAP:
            print(f"Unknown model: {model_name}. Add it to MODEL_CLASS_MAP.")
            continue

        _, model_path = MODEL_CLASS_MAP[model_name]
        print(f"\n{'='*60}")
        print(f"Patching MedRAG for: {model_name}")
        print(f"{'='*60}")

        tokenizer, model = load_model(model_path, args.cache_dir)

        for task in TASKS:
            json_path = os.path.join(
                args.results_dir,
                f"{model_name}_medhalt_reasoning_{task}_seed0.json",
            )
            if not os.path.exists(json_path):
                print(f"  File not found: {json_path}, skipping.")
                continue

            print(f"\n  Task: {task} — {json_path}")
            patch_file(json_path, tokenizer, model, retrieval_system)

        # Unload model before loading next one
        print(f"\nUnloading {model_name} ...")
        del model, tokenizer
        clear_gpu()

    print("\nAll done. Re-run evaluate_results.py to get updated scores.")


if __name__ == "__main__":
    main()
