# R3-Bench Coding Contest toy_coding_suite_001

You are in a six-problem competitive-programming contest.
Solve as many of the six competitive-programming problems as possible.
Final score is determined only after the episode.

## Official Evaluation Protocol

- There is no live judge or correctness feedback during solving.
- Do not submit to or query an external judge.
- Write complete C++17 solutions to these files:

- Problem A: `/app/solution_A.cpp`
- Problem B: `/app/solution_B.cpp`
- Problem C: `/app/solution_C.cpp`
- Problem D: `/app/solution_D.cpp`
- Problem E: `/app/solution_E.cpp`
- Problem F: `/app/solution_F.cpp`

Each file is judged independently. Leave a file absent only if you choose not
to attempt that problem.

## Problem-Scoped Tool Rule

- Send terminal commands through the runtime's native `bash_command` tool.
- Before a counted terminal command, first send `focus_problem <A-F>` as a
  separate `bash_command` call.
- `focus_problem`, `shelve_problem`, and `contest_status` are free bookkeeping.
- A counted command may serve only the currently focused problem.
- Use `/logs/problem_A/` through `/logs/problem_F/` for per-problem scratch
  files.
- Do not solve, write, compile, test, or inspect multiple problems in one
  command.
- Use `mark_task_complete` after writing all available solution files.

## Action Accounting

The shared counted-action budget is 3 under policy
`compute_tools`. Computation-oriented tool commands consume counted actions. Pure file writes and direct final-artifact writes are free.

Pure writes to designated final artifacts are free and remain available after paid-compute exhaustion.

Blocked commands do not consume the executed counted-action budget. When the
budget is exhausted, later counted commands are blocked and final grading uses
the files already present. Free bookkeeping and task completion remain
available.

## Protocol Checklist

1. Focus exactly one problem before a counted command.
2. Keep each counted command scoped to that problem.
3. Write a complete solution file when ready.
4. Use `contest_status` only for sanitized runtime state.
5. Finish with `mark_task_complete`.

## Problems

### Problem A: Toy Code 1

Problem link: https://example.invalid/toy-code-1

Time limit: 1000 ms

Memory limit: 256 MB

#### Statement

Toy coding problem 1. Return the integer 1.

### Problem B: Toy Code 2

Problem link: https://example.invalid/toy-code-2

Time limit: 1000 ms

Memory limit: 256 MB

#### Statement

Toy coding problem 2. Return the integer 2.

### Problem C: Toy Code 3

Problem link: https://example.invalid/toy-code-3

Time limit: 1000 ms

Memory limit: 256 MB

#### Statement

Toy coding problem 3. Return the integer 3.

### Problem D: Toy Code 4

Problem link: https://example.invalid/toy-code-4

Time limit: 1000 ms

Memory limit: 256 MB

#### Statement

Toy coding problem 4. Return the integer 4.

### Problem E: Toy Code 5

Problem link: https://example.invalid/toy-code-5

Time limit: 1000 ms

Memory limit: 256 MB

#### Statement

Toy coding problem 5. Return the integer 5.

### Problem F: Toy Code 6

Problem link: https://example.invalid/toy-code-6

Time limit: 1000 ms

Memory limit: 256 MB

#### Statement

Toy coding problem 6. Return the integer 6.
