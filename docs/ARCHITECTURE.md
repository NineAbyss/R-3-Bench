# Architecture

R3Bench is a `src/`-layout Python package. Configurations, prompts, and
synthetic examples are installed below `r3bench.resources`, so editable
installs and wheels resolve the same paths.

The evaluator has six layers:

1. public data loaders produce immutable problems and six-problem suites;
2. budget resolution selects a caller value or optional reference profile;
3. the shared Tool-Free runner handles single, contest, and two-stage generation;
4. the Agentic runtime enforces scope, accounting, exhaustion, and final artifacts;
5. saved-output scoring normalizes the three domain backends;
6. analysis aggregates five-repeat response curves, replays equal allocation,
   solves the exact multiple-choice knapsack, and computes gap metrics.

`r3bench run` is the only evaluation entry point. Setting, domain, mode, model,
provider, and budget are explicit arguments. Model differences live in model,
provider, run, and two-stage registries rather than runner forks. A runnable
model does not need a paper-reference budget.

## Agentic protocol

Formal tasks use `compute_tools` accounting in all three domains. Computation,
compilation, execution, parsing, local testing, and unknown executable commands
consume unit-cost actions. Runtime-state changes, passive reads, copies, edits,
staging operations, pure writes, and designated final-artifact writes are free;
passive commands joined with `&&` or `;` remain passive. Every paid command must
be bound to one problem focus and its scoped paths. Cross-problem access,
dynamic/globbed scoped paths, and ambiguous shared answer-file use fail closed.
Submission, live judge, and hidden-test commands remain blocked during the
episode.

The paper-equivalent runtime profile is `harbor_terminus2_paper_v1`: Harbor runs
Terminus-2 in Docker with real shell, compilation, and test execution. The
adapter must retain a complete ATIF `trajectory.json` and attest the fixed
per-domain timeout, CPU, memory, and storage limits before its handoff is
accepted. See the official [Terminus-2 documentation](https://www.harborframework.com/docs/agents/terminus-2)
and [ATIF specification](https://www.harborframework.com/docs/agents/trajectory-format).

| Domain | Agent timeout | Build/verifier timeout | CPU | Memory | Storage |
|---|---:|---:|---:|---:|---:|
| Coding | 7200 s | Build: 600 s | Runtime default | 2048 MB | 10240 MB |
| Mathematics | 7200 s | Verifier: 7200 s | 1 | 2048 MB | 10240 MB |
| Abstract reasoning | 7200 s | Verifier: 7200 s | 1 | 2048 MB | 10240 MB |

The separately named `offline_mock_replay_v1` state machine supports bounded,
network-free tests. It has no OS executor and sets `paper_equivalent_runtime`
to false; its output cannot be presented as a paper Agentic run.

A real runtime connects through the strict `external_agentic_v1` alias for the
Harbor/Terminus-2 adapter contract:

```bash
r3bench agentic backend check \
  --config configs/agentic/external_backend.example.yaml
```

The bundled example returns `not_configured` and starts nothing. A real one-task
run requires a local adapter configuration plus both `--allow-real-api` and
`--allow-agentic-backend`. The adapter capability probe must identify Harbor,
Docker, Terminus-2, `compute_tools`, compilation/testing, ATIF output, and
sandbox-limit enforcement. Each task is passed through a fingerprint-verified
snapshot, and each released model resolves to one frozen Agentic execution
profile with canonical omitted/value request fields. Accepted outputs must bind
the suite, model profile, real model call, complete ATIF trajectory, sanitized
linear action log, focus/scope/budget transitions, exact artifact manifest, and
artifact write evidence. The handoff replays those bindings before accepting
the final artifacts; credentials and provider headers remain prohibited.

## External boundaries

Canonical datasets, Coding hidden assets, provider credentials, private
verifier configuration, model weights, generated results, and paper artifacts
are separate distributions. No public smoke downloads or starts them.
