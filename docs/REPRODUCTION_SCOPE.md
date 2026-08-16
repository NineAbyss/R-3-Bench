# Reproduction scope

This release targets protocol reproducibility.

It supports the complete public structure: choosing a domain, setting, model,
and budget; loading frozen six-problem suites; running Pure-NL or Agentic
conditions with five independent repeats; parsing and scoring saved outputs;
building six-level empirical response curves; replaying equal allocation;
solving the exact multiple-choice knapsack Oracle; and computing Contest–Oracle
Gap and Gap Ratio.

The release does not require users to reproduce the paper's exact model
versions, API endpoints, retry/timeout settings, rho budgets, full 300-problem
runs, 50-suite campaigns, or numerical tables. Custom budgets and compatible
models define new benchmark conditions and must be labelled accordingly.

Canonical datasets and Coding verification assets are separate legal
distributions. Hidden tests, historical outputs, paper trees, credentials, and
private runtime configurations are intentionally excluded. Repository source
does not bundle historical trajectories; each authorized real
Harbor/Terminus-2 run instead preserves its complete ATIF `trajectory.json` in
that run's output directory. Offline mock/replay runs do not fabricate one.

The paper Agentic runtime is the external `harbor_terminus2_paper_v1` profile,
not the built-in offline state machine. It uses `compute_tools` in all domains,
keeps pure/final writes free, executes commands in Docker, and validates the
paper's domain-specific time, CPU, memory, and storage limits. The explicitly
named `offline_mock_replay_v1` backend validates interfaces only and is not
paper-equivalent.

Paper-reference budgets remain optional profiles, not evaluator limits. Mock
providers and mock scorers validate interfaces only and do not produce formal
benchmark results.
