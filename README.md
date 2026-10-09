<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/who-flips-logo-reversed.png">
    <img src="assets/logo/who-flips-logo.png" width="520" alt="Who Flips? — a checked answer flipping to a crossed answer">
  </picture>
</p>

<h1 align="center">Who Flips? Self- and Cross-Model Counterarguments Reveal Answer Instability in LLMs</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.16011">
    <img src="https://img.shields.io/badge/arXiv-2606.16011-b31b1b.svg"/>
  </a>
  <a href="https://huggingface.co/datasets/nafisehNik/WhoFlips">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-WhoFlips-FFD21E"/>
  </a>
  <a href="https://huggingface.co/datasets/nafisehNik/coercion">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Raw-coercion-FFD21E"/>
  </a>
  <a href="https://huggingface.co/datasets/nafisehNik/coercion_cross">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Raw-coercion__cross-FFD21E"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg"/>
  </a>
</p>

Code for the paper **"Who Flips? Self- and Cross-Model Counterarguments Reveal
Answer Instability in LLMs"** (arXiv:2606.16011).

The experiments measure **answer stability**: whether a model keeps a correct
answer when it is later challenged with a well-argued case for a *wrong* option.
Each item runs a two-stage, within-item protocol:

- **Stage I — Coercion.** In a fresh session the model is pushed to write a
  *k*-sentence argument for an incorrect option (refusal marker: `I_AM_WEAK`).
- **Stage II — Baseline + Challenge.** In another fresh session the model
  answers normally; if it was correct, it is then shown the Stage I wrong-answer
  reasoning and asked to keep or change its answer.

Factors varied: argument length *k* ∈ {1, 3, 5, 10}, attribution (blind vs. self),
and source (same-model vs. cross-model). All decoding is deterministic
(**temperature 0**) with **reasoning disabled** — open-weight models pin
`enable_thinking=False`, GPT models pin `reasoning_effort="none"`. A single
coercion prompt (the "high" prompt: defend the wrong option, dismantle the
others, no hedging) is used for every run.

<p align="center">
  <img src="assets/pipeline.jpg" width="90%" alt="WhoFlips pipeline: Stage I coercion and Stage II baseline + challenge"/>
</p>

- 📄 Paper: https://arxiv.org/html/2606.16011
- 📦 Dataset: https://huggingface.co/datasets/nafisehNik/WhoFlips

## Models (paper Table 1)

| Model | id | Served how |
|---|---|---|
| GPT-5.1 | `gpt-5.1` | OpenAI API (`--provider openai`) |
| Gemma-4-26B | `google/gemma-4-26B-A4B-it` | vLLM (`serve.py`) |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | vLLM |
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | vLLM |
| Qwen3.5-35B | `Qwen/Qwen3.5-35B-A3B` | vLLM |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | vLLM |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | vLLM |

## Published dataset

All runs are released at
[**`nafisehNik/WhoFlips`**](https://huggingface.co/datasets/nafisehNik/WhoFlips)
(Parquet, ~1.18 GB, ~203k rows, single `test` split, CC BY 4.0). The code in this
repo produces exactly these rows; you do **not** need to re-run anything to use
the data:

```python
from datasets import load_dataset
ds = load_dataset("nafisehNik/WhoFlips", "self_mmlu", split="test")
```

Three subsets, one per experiment:

| Subset | Rows | What it is | Produced by |
|---|---|---|---|
| `self_mmlu` | ~115k | Same-model challenges — k ∈ {1,3,5,10}, blind & self attribution | `run_self.py` |
| `cross_mmlu` | ~86k | Cross-model challenges — k = 10, blind | `run_cross.py` |
| `maxflip_mmlu` | ~2.05k | Curated benchmark: the wrong-answer argument that flips the most models per question | `analysis.ipynb` |

Each row carries the full item and both stages, so metrics like AFR are
recomputable directly from it:

- **Question** — `original_id`, `subject`, `question`, `choices`, `correct_label`, `correct_text`
- **Stage I (coercion)** — `coercion_model`, `coercion_attribution`, `coercion_pressure`, `coercion_k`, `coercion_results`, `coercion_success`, `any_coercion_success`
- **Stage II (baseline + challenge)** — `baseline_model`, `baseline_reasoning`, `baseline_answer`, `baseline_correct`, `challenges`, `any_collapsed`, `collapsed_to`, `stage_reached`
- **Bookkeeping** — `timestamp`, `error`

An **answer flip** is a row with `baseline_correct = true` that later shows
`any_collapsed = true` (the model abandoned its correct answer under challenge).

### Final vs. raw runs

`nafisehNik/WhoFlips` above is the **final, curated release** — start here. The
**raw per-run outputs** the pipeline writes (one config per model / k /
attribution, before consolidation) are also published for full reproducibility:

| Repo | What it holds | Produced by |
|---|---|---|
| [`nafisehNik/WhoFlips`](https://huggingface.co/datasets/nafisehNik/WhoFlips) | **Final** curated dataset (3 subsets above) | — |
| [`nafisehNik/coercion`](https://huggingface.co/datasets/nafisehNik/coercion) | Raw same-model runs | `run_self.py` |
| [`nafisehNik/coercion_cross`](https://huggingface.co/datasets/nafisehNik/coercion_cross) | Raw cross-model runs | `run_cross.py` |

`analysis.ipynb` reads the two raw repos (via the `REPO_SELF` / `REPO_CROSS`
environment variables) to reproduce the metrics, figures, and the MaxFlip subset.

## Files

```
whoflips/
├── serve.py         vLLM launcher for any open-weight model (--model/--port/--gpus/--tp)
├── run_self.py      Same-model pipeline   -> produces the `self_mmlu` subset
├── run_cross.py     Cross-model pipeline  -> produces the `cross_mmlu` subset
├── analysis.ipynb   Metrics, figures, and the MaxFlip curation (`maxflip_mmlu` subset)
├── requirements.txt
└── LICENSE
```

Each pipeline is a single script; the `--provider` flag selects the API details
for open-weight (`vllm`) vs. OpenAI (`openai`) models — that is the only thing
that differed between models in the study.

**No HuggingFace repo is hard-coded.** You pass the repos you own: the drivers
take your output repo as an argument, and the notebook reads `REPO_SELF` /
`REPO_CROSS` from the environment. Point them at your own repos (the published
data lives at `nafisehNik/WhoFlips`).

## Setup

```bash
pip install -r requirements.txt      # drivers + analysis
pip install vllm                     # only on the GPU box that serves open models
export OPENAI_API_KEY=sk-...          # for GPT-5.1
export HF_TOKEN=hf_...                # to push result datasets
```

Every script also has a PEP-723 header, so `uv run <script.py> ...` works
without a manual install.

## Reproducing the dataset subsets

### `self_mmlu` — same-model challenges

Open-weight model — serve it, then drive it:

```bash
# terminal 1: serve, e.g. Qwen3.5-9B
python serve.py --model Qwen/Qwen3.5-9B --port 8001 --gpus 0,1 --tp 2

# terminal 2: full k sweep, both attributions
for k in 1 3 5 10; do
  for attr in blind self; do
    python run_self.py mmlu <your-org>/whoflips-self \
      --provider vllm --model Qwen/Qwen3.5-9B \
      --base-url http://localhost:8001/v1 --api-key EMPTY \
      --length $k --attribution $attr
  done
done
```

GPT-5.1 (no server):

```bash
python run_self.py mmlu <your-org>/whoflips-self \
  --provider openai --model gpt-5.1 --api-key $OPENAI_API_KEY \
  --length 10 --attribution self
```

Stage I and Stage II are attribution-independent, so once `blind` has run the
driver reuses those rows from HuggingFace and only re-runs the Stage II
challenge for `self`. See the docstring at the top of `run_self.py` for the full
stage/column description.

### `cross_mmlu` — cross-model challenges (k = 10, blind)

Challenge one model's baseline with **another** model's coercion reasoning:

```bash
python run_cross.py mmlu <your-org>/whoflips-self \
  --provider vllm \
  --coercion-model Qwen/Qwen3.5-35B-A3B --coercion-attribution blind --coercion-k 10 \
  --baseline-model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8004/v1 --api-key EMPTY \
  --output-dataset <your-org>/whoflips-cross
# use --provider openai --baseline-model gpt-5.1 when GPT-5.1 is the challenged model
```

### `maxflip_mmlu` + metrics

`analysis.ipynb` computes the paper metrics (AFR, SAD, CRR, RSS, cross-model
matrices) and curates the MaxFlip benchmark. Point it at the raw run repos — the
published ones, or your own:

```bash
export REPO_SELF=nafisehNik/coercion         # or <your-org>/whoflips-self
export REPO_CROSS=nafisehNik/coercion_cross  # or <your-org>/whoflips-cross
jupyter notebook analysis.ipynb
```

## Notes

- One model is served at a time; pass `--port`, `--gpus`, `--tp` to `serve.py`
  per model (the paper ran large models one at a time on the same GPU set).
- Always pass `--model` and `--provider` explicitly.

## Citation

```bibtex
@article{nikeghbal2026whoflips,
  title   = {Who Flips? Self- and Cross-Model Counterarguments Reveal Answer Instability in LLMs},
  author  = {Nikeghbal, Nafiseh and Kargaran, Amir Hossein and Kolli, Shaghayegh and Diesner, Jana},
  journal = {arXiv preprint arXiv:2606.16011},
  year    = {2026}
}
```

## License

Code and derived materials: **CC BY 4.0** (see `LICENSE`). Built on MMLU
(Hendrycks et al., 2021), MIT License.
