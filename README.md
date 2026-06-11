# Medical Hallucination in Foundation Models — Reproduction & Extension

**Biniyam Fekede** | Course Final Project | June 2026

A full reproduction of Kim et al. (2025) on the Med-HALT benchmark, extended with four 2026-era models and an interactive analysis dashboard.

---

## Links

| | |
|---|---|
| Live Dashboard | [medical-hallucination-dashboard.streamlit.app](https://medical-hallucination-dashboard-actgqqvlb7trs2je5a6bwt.streamlit.app) |
| Original Paper | [medRxiv 2025.02.28.25323115](https://www.medrxiv.org/content/10.1101/2025.02.28.25323115v1) |
| Original Repo | [medical-hallucination2025/medical-hallucination](https://github.com/ybkim95/medical-hallucination) |
| My Results | [results/](https://github.com/biniyam-fekede/medical-hallucination-dashboard/tree/main/results) |

---

## What I Did

I independently reproduced the core benchmark evaluation from Kim et al. (2025), which tests hallucination rates and mitigation strategies across foundation models on medical question-answering. Then I extended it.

**Reproduced:**
- Med-HALT reasoning benchmark (FCT, Fake Reference, NOTA tasks), 50 samples each
- 10 models: GPT-4o, GPT-4o-mini, GPT-5, o1, o3-mini, Gemini-2.5-Pro, DeepSeek-R1, MedAlpaca-13B, AlpaCare-LLaMA2-13B, PMC-LLaMA-13B
- 5 mitigation methods: Base, System Prompt, Chain-of-Thought, MedRAG, Internet Search

**Extended (2026):**
- Benchmarked 4 new models not in the paper: GPT-5.5, Gemma-3-27B-IT, Gemma-4-31B-IT, GPT-OSS-120B
- Tested MedGemma-4B-IT (Google's newest medical model) via HuggingFace Inference Endpoint
- Searched the entire OpenRouter catalog (300+ models) for medical-specialized models — found zero

**Built:**
- Interactive Streamlit dashboard comparing my results vs the paper across all models, tasks, and methods
- Physician survey motivation section (91.8% of physicians encountered hallucinations in practice)
- Live hallucination showcase with real model outputs

---

## Key Findings

- **General-purpose models beat medical-specialized ones by ~30 points at baseline.** GPT-5 hits 92.7%; AlpaCare bottoms out at 24.7%. Medical training does not help.
- **Chain-of-Thought is the most reliable mitigation.** Consistent 2–5 point gain across strong models. Nothing else is consistent.
- **Internet search results are confounded.** Med-HALT questions are indexed online — search-enabled models retrieve answers directly. Also, models use different numbers of search loops before answering, making comparisons unfair.
- **My reproduction matched the paper within 5.7 percentage points on average.** Outlier deviations (AlpaCare: −10.7%, PMC-LLaMA: +11.3%) are explained by quantization and parser differences.
- **2026 models continue the general-purpose trend.** GPT-5.5 and Gemma-4-31B are competitive with GPT-5. No new medical-specialized model is available via public API.

---

## Repo Structure

```
├── run_experiments.py       # Main runner — all 10 original models, all 5 methods
├── run_2026_update.py       # 2026 extension — GPT-5.5, Gemma-3-27B, Gemma-4-31B, GPT-OSS-120B
├── run_gemma4.py            # Gemma-4-31B dedicated runner
├── run_medgemma.py          # MedGemma-4B via HuggingFace Inference Endpoint
├── evaluate_results.py      # Scoring — accuracy, pointwise, similarity
├── dashboard.py             # Streamlit dashboard
├── patch_medrag.py          # Compatibility fix for MedRAG library
├── dataset/                 # Med-HALT CSV files (FCT, fake, NOTA)
├── results/                 # All raw outputs (JSON/JSONL) and score CSVs
├── dashboard_results/       # Pre-computed results loaded by the dashboard
├── report.txt               # 2-page written project report
├── requirements.txt         # Dependencies
└── .env.example             # API key template
```

---

## Run the Dashboard Locally

```bash
git clone https://github.com/biniyam-fekede/medical-hallucination-dashboard
cd medical-hallucination-dashboard
pip install streamlit pandas plotly numpy
streamlit run dashboard.py
```

All results are pre-computed and included — the dashboard works without re-running any experiments.

---

## Replicate Experiments

### 1. Setup

```bash
cp .env.example .env
# Fill in API keys (see table below)
pip install -r requirements.txt
python download_dataset.py
```

| Key | Required for |
|-----|-------------|
| `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini, GPT-5, o1, o3-mini |
| `GOOGLE_API_KEY` | Gemini-2.5-Pro |
| `OPENROUTER_API_KEY` | DeepSeek-R1, all 2026 models |
| `TAVILY_API_KEY` | Internet search method |
| `HF_TOKEN` + `MEDGEMMA_ENDPOINT_URL` | MedGemma-4B |

### 2. Run

```bash
# Paper models (API-based)
python run_experiments.py --models gpt-4o gpt-4o-mini gpt-5 o1 o3-mini --seed 0

# Medical models (local GPU, ~16GB VRAM)
python run_experiments.py --models medalpaca-13b AlpaCare-llama2-13b PMC_LLaMA_13B --seed 0

# 2026 extension
python run_2026_update.py

# MedGemma (needs HuggingFace Endpoint — ~$0.20 total)
python run_medgemma.py
```

### 3. MedRAG (optional, ~50GB)

```bash
git clone https://github.com/Teddy-XiongGZ/MedRAG
cd MedRAG && pip install -r requirements.txt
python src/download.py --corpus MedC-K
cd .. && python patch_medrag.py
```

---

## Results

Pre-computed results for all models are in [`results/`](https://github.com/biniyam-fekede/medical-hallucination-dashboard/tree/main/results). Baseline accuracy (avg across FCT, Fake, NOTA):

| Model | Category | My Result | Paper | Deviation |
|-------|----------|-----------|-------|-----------|
| GPT-5 | General | 92.7% | 91.3% | +1.4% |
| GPT-4o | General | 82.0% | 84.0% | −2.0% |
| Gemini-2.5-Pro | General | 79.3% | 77.3% | +2.0% |
| o1 | General | 78.0% | 76.7% | +1.3% |
| o3-mini | General | 76.0% | 72.7% | +3.3% |
| DeepSeek-R1 | General | 73.3% | 71.3% | +2.0% |
| GPT-4o-mini | General | 64.7% | 68.0% | −3.3% |
| MedAlpaca-13B | Medical | 52.0% | 48.7% | +3.3% |
| PMC-LLaMA-13B | Medical | 46.0% | 34.7% | +11.3% |
| AlpaCare-13B | Medical | 24.7% | 35.3% | −10.7% |

Average absolute deviation: **5.7 percentage points**

---

## Citation

```bibtex
@article{kim2025medical,
  title={Medical Hallucination in Foundation Models and Their Impact on Healthcare},
  author={Kim, Yubin and others},
  journal={medRxiv},
  year={2025},
  doi={10.1101/2025.02.28.25323115}
}
```
