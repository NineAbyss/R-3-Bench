# Data and scoring

## Canonical data

The data release contains three immutable JSONL datasets:

| Domain | File | Problems | Suites |
|---|---|---:|---:|
| Coding | `coding.jsonl` | 300 | 50 |
| Mathematics | `math.jsonl` | 300 | 50 |
| Abstract Reasoning | `abstract_reasoning.jsonl` | 300 | 50 |

Every suite contains six positions with three easy, two medium, and one hard
problem. The packaged `data_manifest.json` records repository coordinates,
license, paths, and SHA-256 digests. Bundled six-row datasets are synthetic and
require `--toy` or `--relaxed`.

## Saved-output scoring

Generation and scoring are separate. A saved-output row contains a public
`problem_id` and one of `parsed_answer`, `prediction`, or `response_text`.
Every domain emits normalized parse status, judge status, correctness, score,
verdict, evaluator, and scoring mode. Formal outputs also carry the run,
request, stage, repeat, source-condition, and execution identities needed to
join generation to scoring. A scoring summary hashes the exact input snapshot;
formal analysis rejects copied, mixed, stale, or partially replaced result
directories.

```bash
r3bench score \
  --domain coding \
  --data examples/data/coding.jsonl \
  --predictions examples/inputs/scoring/coding_saved_outputs.jsonl \
  --output-dir outputs/coding_scoring \
  --scoring-mode mock --relaxed
```

Scoring modes are:

- `mock`: deterministic interface validation only;
- `dry-run`: parser and scorer-configuration validation;
- `production`: an explicitly configured scorer.

Math uses the configured equivalence judge. Abstract Reasoning uses the pinned
Reasoning Gym scorer. Coding uses LightCPVerifier keyed by public `upstream_id`.
Mock results must not be reported as benchmark scores. Missing or malformed
answers receive zero. A per-problem production judge/verifier exception is
recorded as `judge_error`, receives zero, and does not abort the remaining
suite; an invalid scorer configuration still fails before scoring begins.
Formal analysis rejects a parsed answer that was never judged.

The canonical single-problem contracts are `\\boxed{...}` for Mathematics and
`<answer>...</answer>` for Abstract Reasoning. Tool-Free contest parsing first
requires exactly one complete labeled section for each problem. The unlabeled
fallback is accepted only when there are exactly six complete ordered blocks;
duplicate, mixed, missing, extra, or unclosed sections fail closed. It may then
use the paper's explicit problem-level final-answer-line fallback inside each
accepted section. Agentic Math and Abstract Reasoning are stricter:
`/logs/artifacts/answer.txt` must contain exactly one boxed answer
or ordinary `<answer>...</answer>` tag in the corresponding problem section;
missing or malformed contracts receive zero credit.

The production Math equivalence judge is the paper's DeepSeek V4 Flash model.
It is resolved as a scoring-only provider model and is intentionally separate
from the benchmark model inventory. This does not add it as an evaluated model.

## Coding verifier

The release provides the adapter, readiness checker, asset manifest, and two
clearly separated templates. It does not include services, binaries, hidden
tests, endpoints, or asset roots.

The six-row fail-closed check uses no 300-ID manifest:

```bash
r3bench verifier check \
  --data examples/data/coding.jsonl \
  --config configs/verifiers/lightcpverifier.toy.yaml \
  --output outputs/verifier_toy
```

Expected status is `not_configured`, with no process or network call. For the
canonical data, create an ignored local configuration from
`lightcpverifier.local.example.yaml`, then run:

```bash
r3bench verifier validate-assets \
  --manifest configs/verifiers/coding_assets_manifest.json \
  --data public_data/coding.jsonl \
  --output outputs/coding_asset_validation

r3bench verifier check \
  --data public_data/coding.jsonl \
  --config /path/to/private/lightcpverifier.local.yaml \
  --asset-manifest configs/verifiers/coding_assets_manifest.json \
  --output outputs/verifier_canonical
```

Readiness never starts the service. Production scoring remains post-generation
and exposes no live verdict to the model. Agentic scoring first verifies the
artifact manifest, scoped relative path, regular-file status, and SHA-256
digest. Invalid manifest, path, symlink, or digest evidence fails closed; a
digest-valid artifact containing invalid UTF-8 is retained as a parse failure
and receives zero.

## Output boundary

Scoring and analysis exports may contain normalized run metadata, parsed
answers, judge records, sanitized Agentic accounting, and summaries. They must
not contain credentials, private endpoints or paths, provider headers/request
IDs, hidden tests, asset roots, hidden reasoning, or raw service logs.

An authorized real Harbor/Terminus-2 episode is the sole trajectory exception:
its episode-local directory must retain the complete ATIF `trajectory.json`
required by the protocol. That file remains a generated run artifact, is
subject to the same credential/private-data checks, and is never bundled into
the source or public release. Offline mock/replay runs do not synthesize it.
