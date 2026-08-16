# Quickstart

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_user_acceptance_smoke.py
```

`.[test]` installs only pytest. No default check uses a model API, external
verifier, container runtime, or canonical dataset.

## Data and credentials

Download the separately published canonical data:

```bash
r3bench data fetch --output-dir public_data
r3bench data validate --domain math --source public_data/math.jsonl
```

For real providers, set the credential variable named by the selected profile.
Bundled integrations use `QWEN_API_KEY` or `DEEPSEEK_API_KEY`. Never commit a
populated `.env` file.

## Tool-Free

Single-problem generation:

```bash
r3bench run \
  --setting tool_free --domain coding --mode single_problem \
  --model local-mock --data examples/data/coding.jsonl \
  --output-dir outputs/tool_free_single \
  --output-token-budget 2048 \
  --provider mock --toy --limit-problems 1
```

For a two-stage contest, add:

```bash
--protocol two_stage
```

Stage 1 uses the requested output-token budget. Stage 2 uses the independent
practical cap from the selected two-stage profile and receives only the Stage 1
reasoning/visible trace, never the original problem statements.

`r3bench run` performs five independent repeats by default. Use `--repeat-id N`
to execute one repeat in a distributed workflow, or `--repeats 1` only for a
custom smoke condition. Formal response curves use the six-level official
budget profile; custom curves may use a nondecreasing `--budget-grid`.

## Agentic

```bash
r3bench run \
  --setting agentic --domain coding --mode contest \
  --model local-mock --data examples/data/coding.jsonl \
  --output-dir outputs/agentic_contest \
  --counted-action-budget 10 \
  --provider mock --toy --limit-suites 1
```

Agentic single-task response curves use `--mode response_curve` with a counted
action grid. Ordered levels are retained even when two levels have the same
numeric cap, and formal exports create repeat IDs 1 through 5. Scoring occurs
only after completion and receives final artifacts, not live verifier feedback.

The command above uses the explicit `offline_mock_replay_v1` backend and is only
an interface test. Formal runs require a configured Harbor/Terminus-2 adapter:

```bash
r3bench agentic backend check \
  --config path/to/harbor_terminus2.yaml --probe

r3bench agentic backend run \
  --config path/to/harbor_terminus2.yaml \
  --task-dir outputs/exported_task \
  --output-dir outputs/real_episode \
  --model provider/model \
  --allow-real-api --allow-agentic-backend
```

The adapter invokes Harbor with Docker and `--agent terminus-2`. A successful
episode retains `trajectory.json` in ATIF format alongside the final artifacts
and action log. The paper profile rejects a handoff that does not attest OS
execution capability, compilation/testing support, complete trajectory capture,
and the domain-specific Table 14 sandbox limits.

## Real-provider preview

The same public command handles real providers. Always preview first:

```bash
r3bench run \
  --setting tool_free --domain coding --mode single_problem \
  --model deepseek-chat --data examples/data/coding.jsonl \
  --output-dir outputs/deepseek_preview \
  --output-token-budget 2048 \
  --provider real --run-profile deepseek_coding \
  --dry-run --toy --limit-problems 1
```

Replace `--dry-run` with `--allow-real-api` only for an explicitly authorized,
bounded request. Broader real runs require `--confirm-full-run`.

## Score and analyze

Use `r3bench score` for all three domains, then build response-curve and contest
inputs with `r3bench analysis build-response-curve` and
`build-contest-results`. Finish with `r3bench analysis compare`. Complete
schemas and commands are in the data/scoring and Oracle guides.
