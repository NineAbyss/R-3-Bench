# Response curves and Oracle

Formal R3Bench analysis is deterministic and network-free after generation and
scoring. Each problem is evaluated at the six pressure levels
`0, 0.05, 0.1, 0.2, 0.4, 0.8`, with five independent runs per level. The
stored `budget_level` identifies the pressure level even when two positive
Agentic levels round to the same counted-action cap.

For problem `p` and level `l`, the response curve records the empirical success
rate over the five judged outcomes:

```text
q_p(l) = successful repeats / 5
```

The option cost is the configured budget for level `l`. Provider-reported token
or action usage remains available for diagnostics but is never substituted for
that configured cost.

For each six-problem contest and configured contest budget, analysis computes:

1. the contest score, averaged over five independent contest runs;
2. Equal replay, using the richest response-curve level affordable under
   `floor(contest_budget / 6)` for every problem;
3. the exact multiple-choice knapsack Oracle over all `6^6 = 46,656` level
   assignments;
4. `Contest-Oracle Gap = Oracle score - contest score`;
5. `Gap Ratio = Gap / Oracle score`, or null when Oracle score is zero.

The multiple-choice Oracle may choose a cheaper level with a higher empirical
success rate because sampled response curves are not assumed to be monotonic.
Equal replay does not make that optimization: it always selects the highest
affordable level.

Pure-NL cost is configured output tokens. Agentic cost is configured counted
actions under the cross-domain `compute_tools` policy.

For official Tool-Free cells, the model registry determines the protocol and
cost accounting. Models with an independent reasoning channel must use the
registered two-stage protocol, and only Stage 1 output and reasoning tokens
count toward the response-curve budget. Non-thinking models must use the
one-stage protocol. Analysis derives this rule from the profile; it does not
depend on a caller remembering `--stage1-only`.

## Running formal repeats

`r3bench run` uses five repeats by default. `--repeat-id` runs one selected
repeat for distributed execution, while `--repeats 1` is useful only for smoke
or custom diagnostic conditions. Formal response curves use an official
six-level budget profile or an explicitly supplied six-level grid:

```bash
r3bench run \
  --setting tool_free --domain math --mode response_curve \
  --model deepseek-v4-pro \
  --data public_data/math.jsonl \
  --output-dir outputs/math_curve \
  --budget-profile tool_free_math_deepseek_v4_pro_single_problem_response_curve \
  --provider real --allow-real-api --confirm-full-run
```

The response-curve output hierarchy includes both `level_<n>_budget_<cap>` and
`repeat_<n>`, so rounded duplicate Agentic caps remain distinct. Contest output
uses one `repeat_<n>` directory per complete run.

The zero level follows the same path as every other level: it invokes the
selected backend with a configured metered-resource cap of zero, then parses
and scores the resulting output or artifact. Analysis rejects a synthetic zero
point that has no run and scoring directories.

## Building analysis inputs

Run `r3bench analysis build-response-curve` for every scored level and repeat
and use `--append` to assemble one JSONL file. The builder requires
`budget_level`, `repeat_id`, and a unique execution ID in the run summary;
corresponding CLI flags are optional assertions and cannot fabricate missing
metadata. Run `build-contest-results` for every scored contest repeat in the
same way. A formal input therefore contains 30 observations per problem and
five complete trajectories per contest.

Each condition records a stable `condition_id`, `condition_kind`, configured
`contest_budget`, six-entry `response_curve_grid`, and budget unit. An official
comparison requires the curve, contest, and budget document to bind to the
matching released profiles; custom or legacy rows cannot be mixed into that
cell. Production scoring also binds its input digest and per-problem run,
request, stage, or episode identity to the selected generation directory.

Parser failures, missing answers, and downstream judge/verifier failures are
retained as zero-reward observations, as required by the paper. A parsed output
that was never judged is unresolved and rejected from official analysis.
`--allow-unjudged` remains limited to custom interface diagnostics.

```bash
r3bench analysis compare \
  --response-curve outputs/analysis/response_curve_points.jsonl \
  --contest-results outputs/analysis/contest_results.jsonl \
  --budgets outputs/analysis/budgets.json \
  --output-dir outputs/analysis/comparison
```

Older single-observation inputs remain readable as explicitly non-formal legacy
conditions. They retain their historical binary replay semantics and must not
be reported as paper-reference results.
