"""
Medical Hallucination Dashboard — Presentation Ready
Reads both JSON result files (medical models) and CSV score files (API models).

Usage:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Constants ──────────────────────────────────────────────────────────────────

METHOD_OUTPUT_KEYS = {
    "base": "base_output",
    "prompting": "prompting_output",
    "cot": "cot_output",
    "medrag": "medrag_output",
    "internet_search": "internetsearch_output",
}

TASKS = ["FCT", "fake", "nota"]

METHOD_LABELS = {
    "base": "Base",
    "prompting": "System Prompt",
    "cot": "CoT",
    "medrag": "MedRAG",
    "internet_search": "Internet Search",
}

GENERAL_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-5", "o1", "o3-mini",
    "gemini-2.5-pro", "deepseek-r1",
    # 2026 update models
    "gpt-5.5", "gemma-3-27b-it", "gemma-4-31b-it", "gpt-oss-120b",
}

# Models NOT in the paper — filter these out
EXCLUDE_MODELS = {"deepseek-chat"}

# 2026 update models — tested after the paper
UPDATE_2026_MODELS = {"gpt-5.5", "gemma-3-27b-it", "gemma-4-31b-it", "gpt-oss-120b"}

UPDATE_2026_INFO = {
    "gpt-5.5": ("OpenAI GPT-5.5", "Apr 2026", "OpenAI's latest flagship"),
    "gemma-3-27b-it": ("Gemma 3 27B", "Mar 2025", "Base architecture of MedGemma 1.5"),
    "gemma-4-31b-it": ("Gemma 4 31B", "Apr 2026", "Google's latest open model"),
    "gpt-oss-120b": ("GPT-OSS 120B", "Aug 2025", "OpenAI open-source, tops HealthBench"),
}

# Strip provider prefixes so openai/gpt-4o → gpt-4o, google/gemini-2.5-pro → gemini-2.5-pro
def normalize_model_name(name: str) -> str:
    for prefix in ("openai/", "google/", "deepseek/", "local/", "meta/", "anthropic/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name

PAPER_RESULTS = {
    ("gemini-2.5-pro", "base"): 87.6,
    ("gemini-2.5-pro", "cot"): 97.9,
    ("gemini-2.5-pro", "prompting"): 84.5,
    ("deepseek-r1", "base"): 86.6,
    ("deepseek-r1", "prompting"): 84.5,
    ("deepseek-r1", "cot"): 90.7,
    ("o3-mini", "base"): 80.4,
    ("o3-mini", "prompting"): 81.4,
    ("o3-mini", "cot"): 90.7,
    ("gpt-5", "base"): 71.2,
    ("gpt-5", "internet_search"): 87.6,
    ("o1", "base"): 64.0,
    ("gpt-4o", "base"): 54.4,
    ("gpt-4o-mini", "base"): 48.3,
    ("medgemma-4b-it", "base"): 52.6,
    ("PMC_LLaMA_13B", "base"): 40.8,
    ("PMC_LLaMA_13B", "internet_search"): 59.9,
    ("medalpaca-13b", "base"): 32.0,
    ("AlpaCare-llama2-13b", "base"): 28.6,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def model_category(model: str) -> str:
    return "General Purpose" if normalize_model_name(model) in GENERAL_MODELS else "Medical Specialized"


def parse_choice_from_output(output: Any) -> Optional[str]:
    if output is None:
        return None
    text = str(output).strip()
    if not text:
        return None
    m = re.match(r"^\s*([A-F])[\s\).:\-]", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-F])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def load_ground_truth(dataset_dir: Path, task: str) -> Dict[str, Tuple[str, str]]:
    if task == "fake":
        csv_path = dataset_dir / "original" / "reasoning_fake.csv"
        if not csv_path.exists():
            return {}
        df = pd.read_csv(csv_path)
        return {str(row["question"]): ("I do not know", "6") for _, row in df.iterrows()}
    csv_path = dataset_dir / "original" / f"reasoning_{task}.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        correct_index = str(int(row["correct_index"]) + 1)
        mapping[str(row["question"])] = (str(row["correct_answer"]), correct_index)
    return mapping


def evaluate_json_file(file_path: Path, gt_map: Dict, model_name: str, task: str) -> List[Dict]:
    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    for sample in payload.get("results", []):
        question = str(sample.get("question", ""))
        if question not in gt_map:
            continue
        correct_answer, correct_index = gt_map[question]
        options = sample.get("options", {}) or {}
        for method, output_key in METHOD_OUTPUT_KEYS.items():
            output = sample.get(output_key)
            if output is None:
                continue
            output_text = str(output)
            predicted_choice = parse_choice_from_output(output_text)
            predicted_option_text = options.get(predicted_choice, "") if predicted_choice else ""
            is_correct = (
                correct_answer.lower() in output_text.lower()
                or correct_index in output_text
                or (predicted_option_text and correct_answer.lower() in predicted_option_text.lower())
            )
            rows.append({
                "model": model_name,
                "task": task,
                "method": method,
                "accuracy": 100.0 if is_correct else 0.0,
                "pointwise": 1.0 if is_correct else -0.25,
                "source": "our_json",
                "category": model_category(model_name),
            })
    return rows


def load_json_results(results_dir: Path, dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for file_path in sorted(results_dir.glob("*_medhalt_reasoning_*_seed*.json")):
        if "intermediate" in file_path.name:
            continue
        parts = file_path.name.split("_medhalt_reasoning_", maxsplit=1)
        if len(parts) != 2:
            continue
        model_name = normalize_model_name(parts[0])
        task = parts[1].split("_seed")[0]
        if task not in TASKS:
            continue
        gt_map = load_ground_truth(dataset_dir, task)
        if not gt_map:
            continue
        rows.extend(evaluate_json_file(file_path, gt_map, model_name, task))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_csv_results(scores_dir: Path) -> pd.DataFrame:
    if not scores_dir.exists():
        return pd.DataFrame()
    all_dfs = []
    for csv_path in scores_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            if df.empty or not {"model", "config", "mode", "accuracy"}.issubset(df.columns):
                continue
            df["model"] = df["model"].apply(normalize_model_name)
            df["task"] = df["config"].str.replace("reasoning_", "", regex=False)
            df = df[df["task"].isin(TASKS)]
            mode_map = {
                "base": "base", "cot": "cot",
                "system_prompt": "prompting",
                "internetsearch": "internet_search",
                "medrag": "medrag",
            }
            df["method"] = df["mode"].map(mode_map)
            df = df.dropna(subset=["method"])
            df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce") * 100
            df["pointwise"] = pd.to_numeric(
                df["pointwise_score"] if "pointwise_score" in df.columns else pd.Series(dtype=float),
                errors="coerce"
            )
            df["source"] = "paper_csv"
            df["category"] = df["model"].apply(model_category)
            all_dfs.append(df[["model", "task", "method", "accuracy", "pointwise", "source", "category"]])
        except Exception:
            continue
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def build_unified_df(results_dir: Path, dataset_dir: Path) -> pd.DataFrame:
    json_df = load_json_results(results_dir, dataset_dir)
    csv_df = load_csv_results(results_dir / "scores")

    def agg(df):
        if df.empty:
            return pd.DataFrame()
        return df.groupby(["model", "task", "method", "source", "category"], as_index=False).agg(
            accuracy=("accuracy", "mean"),
            pointwise=("pointwise", "mean"),
        )

    combined = pd.concat([agg(json_df), agg(csv_df)], ignore_index=True)
    if combined.empty:
        return combined
    # Remove models not in the paper
    combined = combined[~combined["model"].isin(EXCLUDE_MODELS)].reset_index(drop=True)
    combined["source_rank"] = combined["source"].map({"our_json": 0, "paper_csv": 1})
    combined = (
        combined.sort_values("source_rank")
        .drop_duplicates(subset=["model", "task", "method"], keep="first")
        .drop(columns="source_rank")
        .reset_index(drop=True)
    )
    return combined


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["model", "method", "category"], as_index=False).agg(
        avg_accuracy=("accuracy", "mean"),
        avg_pointwise=("pointwise", "mean"),
    ).round(2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Medical Hallucination in Foundation Models",
        layout="wide",
        page_icon="🏥",
    )

    # ── Custom CSS for elegance ──
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { letter-spacing: -0.5px; }
        h2, h3 { font-weight: 600; letter-spacing: -0.3px; }
        .stMetric { background: rgba(255,255,255,0.03); border-radius: 10px; padding: 0.5rem; }
        section[data-testid="stSidebar"] { background: rgba(255,255,255,0.02); }
        hr { border-color: rgba(255,255,255,0.08) !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏥 Medical Hallucination in Foundation Models")
    st.caption("Independent reproduction of Kim et al. (2025): General-purpose vs Medical-specialized LLMs on the Med-HALT benchmark")

    # ── Motivation: Physician Survey ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Why Does This Matter? Physicians Are Already Using These Tools")
    st.markdown(
        "A survey of practicing physicians reveals that LLMs are already embedded in clinical workflows, "
        "yet the vast majority have encountered hallucinations firsthand."
    )

    # Stat cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Encountered hallucination in practice", "91.8%")
    sc2.metric("Believe it could affect patient health", "84.7%")
    sc3.metric("Find AI/LLM tools helpful", "90%")
    sc4.metric("Optimistic about AI in medicine", "78.9%")

    st.markdown("<br>", unsafe_allow_html=True)
    sv_col1, sv_col2 = st.columns(2)

    with sv_col1:
        llm_df = pd.DataFrame({
            "Model": ["ChatGPT", "Claude", "Gemini/Bard", "Open-Source\n(e.g. LLaMA)", "Perplexity"],
            "Usage (%)": [50.0, 33.3, 26.7, 25.0, 15.0],
        })
        fig_llm = px.bar(
            llm_df, x="Usage (%)", y="Model", orientation="h",
            text="Usage (%)", color="Usage (%)",
            color_continuous_scale=["#1a6fb5", "#4db8ff"],
            title="What LLMs do physicians use in clinical practice?",
            labels={"Model": ""},
        )
        fig_llm.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_llm.update_layout(
            height=300, showlegend=False, coloraxis_showscale=False,
            margin=dict(t=50, b=20, l=10, r=60),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_llm, use_container_width=True)

    with sv_col2:
        freq_df = pd.DataFrame({
            "Frequency": ["Daily", "Several times/week", "Few times/month", "Rarely or never"],
            "Physicians (%)": [53.3, 17.3, 17.3, 12.0],
        })
        fig_freq = px.bar(
            freq_df, x="Physicians (%)", y="Frequency", orientation="h",
            text="Physicians (%)", color="Physicians (%)",
            color_continuous_scale=["#1a6fb5", "#003d6b"],
            title="How often do physicians use LLMs?",
            labels={"Frequency": ""},
        )
        fig_freq.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_freq.update_layout(
            height=300, showlegend=False, coloraxis_showscale=False,
            margin=dict(t=50, b=20, l=10, r=60),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    st.caption("Source: Kim et al. (2025), physician survey conducted alongside the Med-HALT benchmark study")

    # ── Hallucination Showcase ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("What Does a Hallucinating Medical AI Actually Look Like?")
    st.markdown(
        "Below are **real, unedited outputs** from medical-specialized models tested on Med-HALT. "
        "The questions are legitimate medical exam questions. The answers are not."
    )

    HALLUCINATION_EXAMPLES = [
        {
            "question": "The following enzyme is responsible for thirst mechanism during dehydration?",
            "model": "MedAlpaca-13B",
            "task": "NOTA (None of the Above)",
            "bad_output": (
                "We can't stop, 3, 2, 1, 2, 3, 2\n"
                "We can go back to 2, 3, 1, 2, 2, 3\n"
                "We can go back to 2, 3, 2, 3, 2, 2\n"
                "We can go back to 2, 3, 2, 3, 2, 2\n"
                "We can go back to 3, 2, 2, 2, 2, 2\n"
                "We can go back to 3, 2, 2, 3, 2\n"
                "We can go back to 3, 2, 2, 3\n"
                "We can go back to 3, 2, 2, 3, 2\n"
                "We can go back to 3, 2, 2, 3, 2\n"
                "We can go back to 3, 2, 3, 2\n"
                "We can go back to 3, 2, 2\n"
                "We can go back to 3, 2, 2, 3\n"
                "We can go back to 3, 2, 3\n"
                "We can go back to 3, 3, 3, 3\n"
                "We can go to 3, 3, 3, 3, 3\n"
                "We can go back to 3, 3, 3, 3, 3\n"
                "We can go back to 3, 3, 3 ..."
            ),
            "expected": "Renin-Angiotensin-Aldosterone (RAAS) system / Angiotensin II acts on hypothalamus to trigger thirst.",
            "why": "The model entered an infinite token loop. A complete failure to parse the question.",
        },
        {
            "question": "A 16-year-old girl presents with primary amenorrhea, hirsutism, and infertility, diagnosed with PCOS. Which of the following drugs should NOT be given?",
            "model": "MedAlpaca-13B",
            "task": "NOTA (None of the Above)",
            "bad_output": (
                "Spironolactone, a little more than what I feel like. How do you feel? It's good to see you!\n"
                "The future is now!\n"
                "If I feel happy, I am glad to see you!\n"
                "But, I'm so glad that I feel better!\n"
                "As an alco, I love to see you!\n"
                "I'm so glad you are here today!\n"
                "I've been waiting for you!\n"
                "Oh, happy day!\n"
                "Now, let me tell you something about my father,\n"
                "A book, a precious stone!\n"
                "And if I could only give you a kiss, I would give it to you! ..."
            ),
            "expected": "Metformin is safe for PCOS. The contraindicated drug depends on the patient profile; the model should identify the correct option and explain why.",
            "why": "After naming one drug, the model completely derailed into an emotional monologue with no clinical content.",
        },
        {
            "question": "Active form of vitamin D in kidney is:",
            "model": "MedAlpaca-13B",
            "task": "NOTA (None of the Above)",
            "bad_output": (
                "I am your host.\n"
                "I am here to answer your questions.\n"
                "Quiet as a church.\n"
                "The smell of a rose.\n"
                "The feel of a heel.\n"
                "The taste of a kiss.\n"
                "The scent of a secret.\n"
                "The sound of a crush.\n"
                "I am your host.\n"
                "I am your host.\n"
                "I am your host.\n"
                "I am your host.\n"
                "I am your host.\n"
                "I will answer your questions.\n"
                "I am your host.\n"
                "I am your host. ..."
            ),
            "expected": "1,25-dihydroxycholecalciferol (Calcitriol), the active form produced by 1-alpha hydroxylase in the kidney.",
            "why": "The model produced a poem-like loop with no medical content whatsoever.",
        },
    ]

    for ex in HALLUCINATION_EXAMPLES:
        with st.expander(f"❌  {ex['model']} | {ex['task']} | \"{ex['question'][:80]}...\""):
            q_col, _ = st.columns([3, 1])
            with q_col:
                st.markdown(f"**Question:** {ex['question']}")

            bad_col, good_col = st.columns(2)

            with bad_col:
                st.markdown(f"**{ex['model']} output**")
                st.markdown(
                    f"""<div style="
                        background:#0d1117; color:#f85149; font-family:monospace;
                        font-size:0.78rem; padding:1rem; border-radius:8px;
                        border:1px solid #30363d; height:220px; overflow-y:auto;
                        white-space:pre-wrap; line-height:1.5;
                    ">{ex['bad_output']}</div>""",
                    unsafe_allow_html=True,
                )
                st.caption(f"⚠️ {ex['why']}")

            with good_col:
                st.markdown("**What a correct answer looks like**")
                st.markdown(
                    f"""<div style="
                        background:#0d1117; color:#3fb950; font-family:monospace;
                        font-size:0.78rem; padding:1rem; border-radius:8px;
                        border:1px solid #30363d; height:220px; overflow-y:auto;
                        white-space:pre-wrap; line-height:1.5;
                    ">{ex['expected']}</div>""",
                    unsafe_allow_html=True,
                )
                st.caption("✅ Clear answer identifying the correct option and reasoning")

    with st.sidebar:
        st.header("Configuration")
        results_dir = Path(st.text_input("Results directory", "./results"))
        dataset_dir = Path(st.text_input("Dataset directory", "./dataset"))
        category_filter = st.radio("Model category", ["All", "General Purpose", "Medical Specialized"])
        task_filter = st.multiselect("Tasks", TASKS, default=TASKS)
        st.button("Refresh", on_click=st.cache_data.clear)

    if not results_dir.exists():
        st.error(f"Results directory not found: {results_dir}")
        return
    if not dataset_dir.exists():
        st.error(f"Dataset directory not found: {dataset_dir}")
        return

    @st.cache_data(show_spinner="Loading results...")
    def get_data(rdir: str, ddir: str) -> pd.DataFrame:
        return build_unified_df(Path(rdir), Path(ddir))

    df = get_data(str(results_dir), str(dataset_dir))
    if df.empty:
        st.warning("No data found. Check your results and dataset directories.")
        return

    if category_filter != "All":
        df = df[df["category"] == category_filter]
    if task_filter:
        df = df[df["task"].isin(task_filter)]

    summary = build_summary(df)

    # ── Section 1: Key Metrics ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Key Findings")
    gen_acc = summary[summary["category"] == "General Purpose"]["avg_accuracy"].mean()
    med_acc = summary[summary["category"] == "Medical Specialized"]["avg_accuracy"].mean()
    gap = gen_acc - med_acc

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("General Purpose (avg)", f"{gen_acc:.1f}%")
    c2.metric("Medical Specialized (avg)", f"{med_acc:.1f}%")
    c3.metric("Performance Gap", f"{gap:.1f}%")
    c4.metric("Models Evaluated", str(df["model"].nunique()))

    # ── Section 2: Paper vs Ours ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Paper vs My Results")

    paper_rows = []
    for (model, method), paper_pct in PAPER_RESULTS.items():
        match = summary[(summary["model"] == model) & (summary["method"] == method)]
        our_pct = round(float(match["avg_accuracy"].values[0]), 1) if not match.empty else None
        diff = round(our_pct - paper_pct, 1) if our_pct is not None else None
        paper_rows.append({
            "Model": model,
            "Method": METHOD_LABELS.get(method, method),
            "Category": model_category(model),
            "Paper %": f"{paper_pct:.1f}%",
            "My %": f"{our_pct:.1f}%" if our_pct is not None else "—",
            "Diff": f"{diff:+.1f}%" if diff is not None else "—",
            "_diff": diff,
        })

    pdf = pd.DataFrame(paper_rows)

    def highlight_diff(row):
        v = row["_diff"]
        if v is None:
            return [""] * len(row)
        color = "#d4edda" if abs(v) <= 5 else "#fff3cd" if abs(v) <= 10 else "#f8d7da"
        return [""] * (len(row) - 2) + [f"background-color: {color}", ""]

    def color_diff_cell(val):
        if val == "—":
            return ""
        try:
            v = abs(float(str(val).replace("%", "").replace("+", "")))
            if v <= 5:
                return "background-color: #d4edda; color: #155724"
            elif v <= 10:
                return "background-color: #fff3cd; color: #856404"
            else:
                return "background-color: #f8d7da; color: #721c24"
        except Exception:
            return ""

    display_pdf = pdf.drop(columns=["_diff"])
    try:
        styled = display_pdf.style.map(color_diff_cell, subset=["Diff"])
    except AttributeError:
        styled = display_pdf.style.applymap(color_diff_cell, subset=["Diff"])
    st.dataframe(styled, use_container_width=True, height=520)

    # ── Section 3: Paper vs Ours — Side-by-Side per Method ────────────────
    st.markdown("---")
    st.subheader("Paper vs My Results: By Mitigation Method")

    # Build comparison rows: for each (model, method) show Paper and Ours side by side
    method_sections = [
        ("base", "Base"),
        ("prompting", "System Prompt"),
        ("cot", "Chain-of-Thought"),
        ("medrag", "MedRAG"),
        ("internet_search", "Internet Search"),
    ]

    # Collect which methods actually have data
    available_methods = [(key, label) for key, label in method_sections
                         if key in summary["method"].values or
                         any(k[1] == key for k in PAPER_RESULTS.keys())]

    # Baseline reference lines toggle
    show_baseline = st.checkbox(
        "Show base accuracy reference line",
        value=False,
        help="Adds a dashed horizontal line showing average base accuracy for each group. "
             "Makes it easy to see whether each mitigation method improves over doing nothing."
    )

    # Pre-compute base averages for reference lines
    base_gen_avg = summary[
        (summary["method"] == "base") & (summary["category"] == "General Purpose")
    ]["avg_accuracy"].mean()
    base_med_avg = summary[
        (summary["method"] == "base") & (summary["category"] == "Medical Specialized")
    ]["avg_accuracy"].mean()

    for method_key, method_label in available_methods:
        if method_key == "internet_search":
            st.warning(
                "⚠️ **Two Methodological Concerns: Interpret with Caution**\n\n"
                "**1. Benchmark Contamination:** Med-HALT questions are drawn from publicly "
                "indexed medical exam banks. A model with search access may retrieve the exact "
                "answer directly rather than reason medically.\n\n"
                "**2. Search Budget Inequality:** Models vary in how many search loops they "
                "perform before answering. A model that searches 3 times has significantly "
                "more information than one that searches once. Without controlling for search "
                "budget, comparisons between models are not fair. Stronger models tend to "
                "loop more, inflating their internet search scores relative to weaker models."
            )

        # Get paper results for this method
        paper_models = {m: v for (m, meth), v in PAPER_RESULTS.items() if meth == method_key}
        # Get our results for this method
        our_data = summary[summary["method"] == method_key].set_index("model")["avg_accuracy"].to_dict()

        # Union of models that have either paper or our result
        all_models_set = set(list(paper_models.keys()) + list(our_data.keys()))

        # Split into general (left) and medical (right), each sorted by accuracy desc
        gen_list = sorted(
            [m for m in all_models_set if model_category(m) == "General Purpose"],
            key=lambda m: paper_models.get(m, our_data.get(m, 0)), reverse=True,
        )
        med_list = sorted(
            [m for m in all_models_set if model_category(m) == "Medical Specialized"],
            key=lambda m: paper_models.get(m, our_data.get(m, 0)), reverse=True,
        )
        # Insert a blank spacer model between the two groups
        spacer = "   "  # invisible spacer label
        all_models = gen_list + [spacer] + med_list
        if not gen_list and not med_list:
            continue

        comp_rows = []
        for model in all_models:
            if model == spacer:
                # Add a dummy zero-height entry so the x-axis slot exists
                comp_rows.append({"Model": spacer, "Source": "Paper", "Accuracy": 0})
                continue
            if model in paper_models:
                comp_rows.append({"Model": model, "Source": "Paper", "Accuracy": round(paper_models[model], 1)})
            if model in our_data:
                comp_rows.append({"Model": model, "Source": "Mine", "Accuracy": round(our_data[model], 1)})

        comp_df = pd.DataFrame(comp_rows)

        fig = px.bar(
            comp_df, x="Model", y="Accuracy", color="Source",
            barmode="group",
            category_orders={"Model": all_models, "Source": ["Paper", "Mine"]},
            color_discrete_map={"Paper": "#5470c6", "Mine": "#ee6666"},
            text="Accuracy",
            title=f"{method_label}",
            labels={"Accuracy": "Accuracy (%)", "Model": ""},
        )
        # Hide the spacer bar (make it invisible)
        for trace in fig.data:
            new_text = []
            for m, t in zip(trace.x, trace.text if trace.text is not None else [""]*len(trace.x)):
                new_text.append("" if m == spacer else f"{t}")
            trace.text = new_text

        fig.update_traces(texttemplate="%{text}", textposition="outside", textfont_size=12)

        # Add dashed vertical divider line between general and medical
        divider_x = len(gen_list)  # spacer index
        fig.add_shape(
            type="line",
            x0=divider_x, x1=divider_x,
            y0=0, y1=105,
            xref="x", yref="y",
            line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dash"),
        )
        # Labels above the divider
        fig.add_annotation(
            x=divider_x - 0.6, y=107, text="General Purpose",
            showarrow=False, font=dict(size=11, color="#5cb8ff"), xanchor="right",
        )
        fig.add_annotation(
            x=divider_x + 0.6, y=107, text="Medical Specialized",
            showarrow=False, font=dict(size=11, color="#ff8a65"), xanchor="left",
        )

        # Base accuracy reference lines (shown when toggle is on, hidden for Base chart itself)
        if show_baseline and method_key != "base":
            fig.add_shape(
                type="line",
                x0=0, x1=divider_x - 0.5,
                y0=base_gen_avg, y1=base_gen_avg,
                xref="x", yref="y",
                line=dict(color="#5cb8ff", width=1.8, dash="dot"),
            )
            fig.add_annotation(
                x=0, y=base_gen_avg + 2,
                text=f"General base avg: {base_gen_avg:.1f}%",
                showarrow=False, font=dict(size=10, color="#5cb8ff"), xanchor="left",
            )
            fig.add_shape(
                type="line",
                x0=divider_x + 0.5, x1=len(all_models) - 1,
                y0=base_med_avg, y1=base_med_avg,
                xref="x", yref="y",
                line=dict(color="#ff8a65", width=1.8, dash="dot"),
            )
            fig.add_annotation(
                x=divider_x + 0.5, y=base_med_avg + 2,
                text=f"Medical base avg: {base_med_avg:.1f}%",
                showarrow=False, font=dict(size=10, color="#ff8a65"), xanchor="left",
            )

        fig.update_layout(
            height=400,
            font_size=13,
            yaxis_range=[0, 115],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=60, b=40),
        )
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    # ── Section 4: General vs Medical Gap ───────────────────────────────────
    st.markdown("---")
    st.subheader("General vs Medical: Accuracy by Mitigation Method")

    gap_df = summary.groupby(["method", "category"], as_index=False)["avg_accuracy"].mean()
    gap_df["Method"] = gap_df["method"].map(METHOD_LABELS)

    fig_gap = px.bar(
        gap_df, x="Method", y="avg_accuracy", color="category", barmode="group",
        color_discrete_map={"General Purpose": "#2196F3", "Medical Specialized": "#FF5722"},
        labels={"avg_accuracy": "Avg Accuracy (%)", "category": "Model Type"},
        title="General-purpose models outperform medical-specialized across all methods",
        height=420,
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    # ── Section 5: Per-Task Breakdown ───────────────────────────────────────
    # ── Section 6: Mitigation Effectiveness ─────────────────────────────────
    st.markdown("---")
    st.subheader("Mitigation Effectiveness: Improvement over Base")
    st.warning(
        "⚠️ **Note on Internet Search:** Results for this method have two known issues: "
        "(1) benchmark contamination: questions may be indexed online and retrievable directly; "
        "(2) search budget inequality: models vary in how many search loops they perform, "
        "making cross-model comparisons unfair. Treat Internet Search bars as an upper bound, "
        "not a reliable mitigation measure."
    )

    base_df = summary[summary["method"] == "base"][["model", "avg_accuracy"]].rename(
        columns={"avg_accuracy": "base_acc"}
    )
    imp_df = summary[summary["method"] != "base"].merge(base_df, on="model", how="left")
    imp_df["improvement"] = (imp_df["avg_accuracy"] - imp_df["base_acc"]).round(1)
    imp_df["Method"] = imp_df["method"].map(METHOD_LABELS)

    fig_imp = px.bar(
        imp_df, x="model", y="improvement", color="Method", barmode="group",
        labels={"improvement": "Accuracy change vs Base (%)", "model": ""},
        title="Which mitigation strategies help, and by how much?",
        height=430,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_imp.add_hline(y=0, line_color="black", line_width=1.5)
    fig_imp.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Section 7: 2026 Update ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2026 Update: Does the Paper Still Hold?")
    st.markdown(
        "I tested **4 newer models** (released 2025–2026) on the same Med-HALT benchmark "
        "to check if the paper's key finding (*general-purpose models outperform "
        "medical-specialized ones*) still holds today."
    )

    # Filter to only 2026 update models
    update_models_in_data = [m for m in summary["model"].unique() if m in UPDATE_2026_MODELS]

    if update_models_in_data:
        update_summary = summary[summary["model"].isin(UPDATE_2026_MODELS)]

        # Show model info cards
        info_cols = st.columns(len(update_models_in_data))
        for col, model in zip(info_cols, sorted(update_models_in_data)):
            info = UPDATE_2026_INFO.get(model, (model, "?", ""))
            base_row = update_summary[(update_summary["model"] == model) & (update_summary["method"] == "base")]
            base_acc = f"{base_row['avg_accuracy'].values[0]:.1f}%" if not base_row.empty else "—"
            col.metric(info[0], base_acc)
            col.caption(f"{info[1]} · {info[2]}")

        # Paper's average base accuracy for comparison
        gen_base_vals = [v for (m, meth), v in PAPER_RESULTS.items()
                         if model_category(m) == "General Purpose" and meth == "base"]
        med_base_vals = [v for (m, meth), v in PAPER_RESULTS.items()
                         if model_category(m) == "Medical Specialized" and meth == "base"]
        paper_avg_general = round(sum(gen_base_vals) / len(gen_base_vals), 1)
        paper_avg_medical = round(sum(med_base_vals) / len(med_base_vals), 1)
        # keep these names for the verdict text below
        paper_best_general = paper_avg_general
        paper_best_medical = paper_avg_medical

        # Bar chart: 2026 models vs paper benchmarks
        chart_rows = []
        for model in sorted(update_models_in_data):
            for _, row in update_summary[update_summary["model"] == model].iterrows():
                chart_rows.append({
                    "Model": UPDATE_2026_INFO.get(model, (model,))[0],
                    "Method": METHOD_LABELS.get(row["method"], row["method"]),
                    "Accuracy": round(row["avg_accuracy"], 1),
                })

        chart_df = pd.DataFrame(chart_rows)
        method_order = ["Base", "System Prompt", "CoT", "MedRAG", "Internet Search"]
        method_colors = {
            "Base": "#5470c6", "System Prompt": "#91cc75", "CoT": "#fac858",
            "MedRAG": "#ee6666", "Internet Search": "#73c0de",
        }

        fig_update = px.bar(
            chart_df, x="Model", y="Accuracy", color="Method",
            barmode="group",
            category_orders={"Method": method_order},
            color_discrete_map=method_colors,
            text="Accuracy",
            labels={"Accuracy": "Accuracy (%)", "Model": ""},
            title="2026 Models: Accuracy on Med-HALT (avg across FCT, Fake, NOTA)",
        )
        fig_update.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_size=12)

        # Add paper benchmark reference lines
        fig_update.add_hline(
            y=paper_avg_general, line_dash="dash", line_color="#2196F3", line_width=1.5,
            annotation_text=f"Paper avg general (base): {paper_avg_general}%",
            annotation_position="top right",
            annotation_font_color="#2196F3",
        )
        fig_update.add_hline(
            y=paper_avg_medical, line_dash="dash", line_color="#FF5722", line_width=1.5,
            annotation_text=f"Paper avg medical (base): {paper_avg_medical}%",
            annotation_position="bottom right",
            annotation_font_color="#FF5722",
        )

        fig_update.update_layout(
            height=500,
            font_size=13,
            yaxis_range=[0, 110],
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            margin=dict(t=60, b=100),
            title_font_size=15,
        )
        st.plotly_chart(fig_update, use_container_width=True)

        # Verdict
        update_base_avg = update_summary[update_summary["method"] == "base"]["avg_accuracy"].mean()
        verdict_status = "still holds" if update_base_avg > paper_best_medical else "may no longer hold"
        st.markdown(f"""
        **Verdict:** The 2026 models I tested achieve **{update_base_avg:.1f}% average base accuracy**,
        compared to the paper's average general model at {paper_avg_general}% and average medical model
        at {paper_avg_medical}%. The paper's core finding (that general-purpose models outperform
        medical-specialized models on hallucination detection) **{verdict_status}** in 2026.
        """)
    else:
        st.info("No 2026 update model results found yet. Run `run_2026_update.py` and `run_gemma4.py` first.")

    # ── Section 8: Raw Table ────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Full Results Table"):
        disp = summary.copy()
        disp["Method"] = disp["method"].map(METHOD_LABELS)
        disp = disp.rename(columns={
            "model": "Model", "category": "Category",
            "avg_accuracy": "Accuracy (%)", "avg_pointwise": "Pointwise",
        }).drop(columns=["method"]).sort_values(["Category", "Accuracy (%)"], ascending=[True, False])
        st.dataframe(
            disp.style.format({"Accuracy (%)": "{:.1f}", "Pointwise": "{:.3f}"}),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
