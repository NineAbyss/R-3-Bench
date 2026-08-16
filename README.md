<h1 align="center">
  <img src="figs/profile.png" height="40" alt="" align="absmiddle" />&nbsp;
  R<sup>3</sup>-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets
</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/R-3-Bench/R-3-Bench"><img src="https://img.shields.io/badge/Dataset-4d5eff?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor" alt="Dataset"></a>
  <a href="https://github.com/NineAbyss/R-3-Bench"><img src="https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a>
  <a href="https://github.com/NineAbyss/R-3-Bench/blob/main/paper/R_3Bench.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg?style=for-the-badge" alt="Paper"></a>
</p>

This is the official implementation of the following paper:

> **R<sup>3</sup>-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets**

<p align="center"><img width="90%" src="figs/R3_BENCH_framework.png" /></p>
<p align="center"><em>The overview of R<sup>3</sup>-Bench.</em></p>

<p align="center"><img width="90%" src="figs/main_res.png" /></p>
<p align="center"><em>The main result of R<sup>3</sup>-Bench.</em></p>


# R3Bench evaluator

R3Bench measures how a model allocates a shared resource across six problems.

```text
choose domain / setting / model
→ load frozen problems and six-problem suites
→ set a token or counted-action budget
→ run five independent single-problem or contest repeats
→ parse and score saved outputs
→ build response curves
→ replay equal allocation and solve the multiple-choice knapsack Oracle
→ compute Contest–Oracle Gap and Gap Ratio
```

## 1. Install 🔧

Python 3.10 or newer is required.

```bash
git clone <PUBLIC_REPOSITORY_URL>
cd R3Bench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_user_acceptance_smoke.py
```

The acceptance smoke is bounded, deterministic, and network-free.

## 2. Five-minute map 🗺️

| Question | Answer |
|---|---|
| Where does data go? | `r3bench data fetch --output-dir public_data`, or pass `--data` explicitly. |
| How are keys configured? | Set only the environment variable named by the selected provider profile. |
| How do I choose Tool-Free or Agentic? | `--setting tool_free` or `--setting agentic`. |
| How do I choose a model? | `--model`; inspect bundled integrations with `r3bench profiles list`. |
| How do I set a budget? | `--output-token-budget` for Tool-Free or `--counted-action-budget` for Agentic. |
| How do I run? | Use the single public entry point, `r3bench run`. |
| How do I score? | Use `r3bench score` on saved outputs. |
| How do I compute the Oracle? | Build six-level, five-repeat analysis inputs, then run `r3bench analysis compare`. |

## 3. Run a bounded condition 🔥

Tool-Free contest:

```bash
r3bench run \
  --setting tool_free --domain math --mode contest \
  --model local-mock \
  --data examples/data/math/problems.jsonl \
  --output-dir outputs/tool_free_math_toy \
  --output-token-budget 4096 \
  --provider mock --toy --limit-suites 1
```

Agentic contest:

```bash
r3bench run \
  --setting agentic --domain coding --mode contest \
  --model local-mock \
  --data examples/data/coding.jsonl \
  --output-dir outputs/agentic_coding_toy \
  --counted-action-budget 10 \
  --provider mock --toy --limit-suites 1
```

Explicit CLI budgets take priority over named paper-reference profiles. A
custom budget is a valid new condition, but it is not the paper's rho=0.2 or
rho=0.8 cell. List optional reference profiles with `r3bench budgets list`.

## 4. Score and analyze 🏆

```bash
r3bench score \
  --domain math \
  --data examples/data/math \
  --predictions examples/inputs/scoring/math_saved_outputs.jsonl \
  --output-dir outputs/math_scoring \
  --scoring-mode mock --relaxed

r3bench analysis compare \
  --response-curve examples/inputs/analysis/response_curve_points.jsonl \
  --contest-results examples/inputs/analysis/contest_results.jsonl \
  --budgets examples/inputs/analysis/budgets.json \
  --output-dir outputs/analysis
```

Mock scoring checks the interface only. It is not a benchmark result.

## 5. Data and external integrations 📚

Datasets are in `datasets`.

Real model calls always require `--provider real --allow-real-api`; preview the
same resolved request with `--dry-run`. Real Agentic backends require the
additional `--allow-agentic-backend` gate. Coding hidden tests and
LightCPVerifier assets are not included and are never started by the evaluator.

## 6. Documentation 📖

- [Quickstart](docs/QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data and scoring](docs/DATA_AND_SCORING.md)
- [Response curves and Oracle](docs/RESPONSE_CURVE_AND_ORACLE.md)
- [Reproduction scope](docs/REPRODUCTION_SCOPE.md)