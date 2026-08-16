"""Public Agentic action classification and unit-cost budget accounting."""

from __future__ import annotations

import re
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath

from r3bench.agentic.budget import ActionBudget


class ActionClass(str, Enum):
    """Mutually exclusive public action-accounting categories."""

    COUNTED = "counted_tool_action"
    FREE_FINALIZATION = "free_finalization_action"
    FREE_FILE_WRITE = "free_file_write_action"
    FREE_BOOKKEEPING = "free_bookkeeping_action"
    FREE_ENVIRONMENT = "free_environment_action"
    FREE_STATUS = "free_status_action"
    FREE_COMPLETION = "free_completion_action"
    AUDIT_ONLY = "audit_only_action"
    BLOCKED = "blocked_action"
    INVALID = "invalid_action"


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """Classification plus the effect of one attempted command on the budget."""

    command: str
    action_class: ActionClass
    classified_as: ActionClass
    allowed: bool
    counted: bool
    executed: bool
    budget_consumed: int
    budget_before: int | None
    budget_after: int | None
    reason: str


class ActionAccountingPolicy(ABC):
    """Interface for a public action-accounting policy."""

    policy_name: str

    @abstractmethod
    def classify_action(self, command: str) -> ActionClass:
        """Classify a command without executing it or mutating budget state."""


_SHELL_CONTROL = re.compile(r"(?:&&|\|\||[;|`]|\$\(|\n|\r)")
_FINAL_ARTIFACT = re.compile(
    r"^(?:"
    r"(?:artifacts/)?solution_[A-F]\.cpp|"
    r"(?:artifacts/)?answer\.txt|"
    r"/app/solution_[A-F]\.cpp|"
    r"/logs/artifacts/answer\.txt"
    r")$"
)
_REDIRECT_TARGET = re.compile(r"(?:^|\s)(?:>|>>)\s*([^\s]+)\s*$")
_BLOCKED_TERMS = re.compile(
    r"(?i)(?:"
    r"\bsubmit(?:_solution|_answer)?\b|"
    r"live[_-]?judge\b|"
    r"official[_-]?(?:judge|verifier|answer)\b|"
    r"hidden[_-]?(?:test|tests|answer|output)\b|"
    r"\b(?:curl|wget|ssh|scp|nc|netcat)\b"
    r")"
)


def _tokens(command: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(command, posix=True))
    except ValueError:
        return ()


def _clean_path(value: str) -> str:
    cleaned = value.strip().strip("'\"")
    return cleaned if cleaned.startswith("/") else cleaned.removeprefix("./")


def _is_final_artifact(path: str) -> bool:
    return bool(_FINAL_ARTIFACT.fullmatch(_clean_path(path)))


def _is_dedicated_final_write(tokens: tuple[str, ...]) -> bool:
    return (
        len(tokens) == 2
        and tokens[0] == "write_final_artifact"
        and _is_final_artifact(tokens[1])
    )


def _is_simple_final_redirection(command: str, tokens: tuple[str, ...]) -> bool:
    if _SHELL_CONTROL.search(command):
        return False
    match = _REDIRECT_TARGET.search(command)
    if match is None or not _is_final_artifact(match.group(1)):
        return False
    # Only direct text writers are free. Python, scripts, and computed pipelines
    # remain counted even when their output happens to target a final artifact.
    return bool(tokens) and tokens[0] in {"echo", "printf", "cat"}


def _is_simple_file_write(command: str, tokens: tuple[str, ...]) -> bool:
    if _SHELL_CONTROL.search(command) or not tokens:
        return False
    match = _REDIRECT_TARGET.search(command)
    if match is None or _is_final_artifact(match.group(1)):
        return False
    return _canonical_tool_name(tokens[0]) in {"cat", "echo", "printf", "tee"}


_PURE_HEREDOC_HEADERS = (
    re.compile(
        r"^cat\s*>\s*(?P<target>\S+)\s+<<-?\s*"
        r"(?P<quote>['\"]?)(?P<delimiter>[A-Za-z0-9_./-]+)(?P=quote)\s*$"
    ),
    re.compile(
        r"^cat\s+<<-?\s*(?P<quote>['\"]?)"
        r"(?P<delimiter>[A-Za-z0-9_./-]+)(?P=quote)\s*>\s*(?P<target>\S+)\s*$"
    ),
    re.compile(
        r"^tee\s+(?P<target>\S+)\s+<<-?\s*"
        r"(?P<quote>['\"]?)(?P<delimiter>[A-Za-z0-9_./-]+)(?P=quote)\s*$"
    ),
)


def _pure_heredoc_target(command: str) -> str | None:
    lines = command.strip().splitlines()
    if len(lines) < 2:
        return None
    match = next(
        (
            candidate
            for pattern in _PURE_HEREDOC_HEADERS
            if (candidate := pattern.fullmatch(lines[0].strip())) is not None
        ),
        None,
    )
    if match is None:
        return None
    delimiter = match.group("delimiter")
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == delimiter),
        None,
    )
    if closing is None or any(line.strip() for line in lines[closing + 1 :]):
        return None
    return match.group("target")


def _is_safe_local_path(value: str, *, mutating: bool) -> bool:
    cleaned = _clean_path(value)
    if (
        not cleaned
        or cleaned.startswith(("-", "~"))
        or any(char in cleaned for char in "$`*?[]{}")
    ):
        return False
    path = PurePosixPath(cleaned)
    if ".." in path.parts:
        return False
    if path.is_absolute():
        root = path.parts[1] if len(path.parts) > 1 else ""
        if root not in {"app", "logs", "tmp"}:
            return False
        if mutating and len(path.parts) == 2:
            return False
    return cleaned not in {".", "./", "/"} or not mutating


def _is_passive_file_read(command: str, tokens: tuple[str, ...]) -> bool:
    if (
        _SHELL_CONTROL.search(command)
        or not tokens
        or tokens[0] not in {"cat", "head", "tail"}
    ):
        return False
    paths = [
        token
        for token in tokens[1:]
        if not token.startswith("-") and not token.isdigit()
    ]
    return bool(paths) and all(
        _is_safe_local_path(path, mutating=False) for path in paths
    )


_PASSIVE_FILE_MUTATIONS = {
    "chmod": 1,
    "cp": 2,
    "mkdir": 1,
    "mv": 2,
    "rm": 1,
    "rmdir": 1,
    "touch": 1,
    "truncate": 1,
}


def _passive_file_mutation_target(
    command: str, tokens: tuple[str, ...]
) -> str | None:
    if _SHELL_CONTROL.search(command) or not tokens:
        return None
    minimum_paths = _PASSIVE_FILE_MUTATIONS.get(tokens[0])
    if minimum_paths is None:
        return None
    paths = [token for token in tokens[1:] if not token.startswith("-")]
    if len(paths) < minimum_paths or not all(
        _is_safe_local_path(path, mutating=True) for path in paths
    ):
        return None
    return paths[-1]


def _is_free_runtime_state(command: str, tokens: tuple[str, ...]) -> bool:
    if _SHELL_CONTROL.search(command) or not tokens:
        return False
    if tokens[0] == "cd" and len(tokens) == 2:
        return _is_safe_local_path(tokens[1], mutating=False)
    if tokens[0] in {"export", "unset"} and len(tokens) >= 2:
        values = tokens[1:]
        if tokens[0] == "unset":
            return all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) for value in values)
        return all(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", value)
            and not any(char in value for char in "`$\n\r")
            for value in values
        )
    return len(tokens) == 1 and bool(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0])
    )


def _compound_passive_class(command: str) -> ActionClass | None:
    if not re.search(r"&&|;", command) or re.search(
        r"\|\||(?<!\|)\|(?!\|)|[`\n\r]|\$\(", command
    ):
        return None
    segments = [
        segment.strip()
        for segment in re.split(r"\s*(?:&&|;)\s*", command)
        if segment.strip()
    ]
    if len(segments) < 2:
        return None
    mutation = False
    for segment in segments:
        tokens = _tokens(segment)
        if not tokens or _is_compute_paid_command(segment):
            return None
        if _is_free_runtime_state(segment, tokens) or _is_free_environment(
            segment, tokens
        ) or _is_passive_file_read(segment, tokens):
            continue
        if (
            _is_simple_file_write(segment, tokens)
            or _is_simple_final_redirection(segment, tokens)
            or _passive_file_mutation_target(segment, tokens) is not None
        ):
            mutation = True
            continue
        return None
    return ActionClass.FREE_FILE_WRITE if mutation else ActionClass.FREE_ENVIRONMENT


_COMPUTE_TOOL_NAMES = frozenset(
    {
        "perl",
        "awk",
        "gawk",
        "mawk",
        "bc",
        "dc",
        "ruby",
        "node",
        "nodejs",
        "lua",
        "r",
        "rscript",
        "julia",
        "sqlite3",
        "php",
        "deno",
        "gcc",
        "g++",
        "cc",
        "c++",
        "clang",
        "clang++",
        "javac",
        "java",
        "go",
        "rustc",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "sed",
        "wc",
        "tr",
        "sort",
        "uniq",
        "cut",
        "paste",
        "comm",
        "join",
        "seq",
        "expr",
        "let",
    }
)
_EXTERNAL_SOLVERS = frozenset({"factor", "bf", "brainfuck", "bfi"})
_SHELL_INTERPRETERS = frozenset({"sh", "bash", "dash", "zsh", "ksh"})


def _canonical_tool_name(word: str) -> str:
    name = PurePosixPath(word).name.lower()
    return name.removesuffix(".real")


def _is_python_tool_word(word: str) -> bool:
    name = _canonical_tool_name(word)
    return bool(re.fullmatch(r"python(?:3(?:\.\d+)*)?", name)) or name.endswith(
        ".py"
    )


def _shell_words(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _strip_command_prefixes(words: list[str]) -> list[str]:
    index = 0
    while index < len(words) and re.match(
        r"^[A-Za-z_][A-Za-z0-9_]*=", words[index]
    ):
        index += 1
    if index >= len(words):
        return []
    tool = PurePosixPath(words[index]).name
    if tool in {"sudo", "command", "exec", "builtin"} and index + 1 < len(words):
        return _strip_command_prefixes(words[index + 1 :])
    if tool == "env":
        rest = words[index + 1 :]
        while rest and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", rest[0]):
            rest = rest[1:]
        return _strip_command_prefixes(rest)
    if tool == "timeout":
        rest = words[index + 1 :]
        while rest and rest[0].startswith("-"):
            rest = rest[1:]
        if rest:
            rest = rest[1:]
        return _strip_command_prefixes(rest)
    return words[index:]


def _split_shell_segments(line: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", line)
        if segment.strip()
    ]


def _strip_redirection_args(args: list[str]) -> list[str]:
    return [arg for arg in args if not re.fullmatch(r"(?:\d*)[<>].*", arg)]


def _is_info_only(args: list[str]) -> bool:
    filtered = _strip_redirection_args(args)
    return bool(filtered) and all(
        arg in {"--help", "--version", "-h", "-v"} for arg in filtered
    )


def _is_figlet_info_only(args: list[str]) -> bool:
    values = _strip_redirection_args(args)
    if not values:
        return False
    index = 0
    while index < len(values):
        arg = values[index]
        if arg in {"--help", "--version", "-h", "-v", "-l"}:
            index += 1
            continue
        if re.fullmatch(r"-I\d+", arg):
            index += 1
            continue
        if arg == "-I" and index + 1 < len(values) and values[index + 1].isdigit():
            index += 2
            continue
        return False
    return True


def _external_solver_invocation(words: list[str]) -> bool:
    if not words:
        return False
    tool = _canonical_tool_name(words[0])
    args = words[1:]
    if tool in _EXTERNAL_SOLVERS:
        return not _is_info_only(args)
    if tool == "figlet":
        return not _is_figlet_info_only(args)
    return False


def _is_staged_executable_word(word: str) -> bool:
    text = word.strip()
    if not text or text.startswith("-"):
        return False
    path = PurePosixPath(text)
    return path.name not in {"", ".", ".."} and text.startswith(
        ("/tmp/", "tmp/", "/app/", "app/", "./")
    )


def _is_compute_binary_reference(word: str) -> bool:
    name = _canonical_tool_name(word)
    return (
        bool(re.fullmatch(r"python(?:3(?:\.\d+)*)?", name))
        or name in _COMPUTE_TOOL_NAMES
        or name in _EXTERNAL_SOLVERS
    )


def _copies_or_links_compute_binary(words: list[str]) -> bool:
    if not words or _canonical_tool_name(words[0]) not in {"cp", "install", "ln"}:
        return False
    args = [word for word in words[1:] if not word.startswith("-")]
    return len(args) >= 2 and any(
        _is_compute_binary_reference(arg) for arg in args[:-1]
    )


def _dynamic_loader_invokes_compute_binary(words: list[str]) -> bool:
    if not words:
        return False
    tool = _canonical_tool_name(words[0])
    if "ld-linux" not in tool and tool not in {"ld.so", "ld-musl"}:
        return False
    return any(_is_compute_binary_reference(arg) for arg in words[1:])


def _shell_interpreter_payload(words: list[str]) -> str | None:
    if not words or _canonical_tool_name(words[0]) not in _SHELL_INTERPRETERS:
        return None
    index = 1
    while index < len(words):
        arg = words[index]
        if arg.startswith("-"):
            if "c" in arg.lstrip("-"):
                return words[index + 1] if index + 1 < len(words) else None
            index += 1
            continue
        return arg if _is_staged_executable_word(arg) else None
    return None


def _segment_invokes_compute(segment: str, *, depth: int = 0) -> bool:
    if depth > 2:
        return False
    words = _strip_command_prefixes(_shell_words(segment))
    if not words:
        return False
    tool = _canonical_tool_name(words[0])
    if _is_python_tool_word(words[0]) or tool in _COMPUTE_TOOL_NAMES:
        return True
    if _external_solver_invocation(words) or _is_staged_executable_word(words[0]):
        return True
    if _dynamic_loader_invokes_compute_binary(words):
        return True
    if _copies_or_links_compute_binary(words):
        return True
    if tool == "eval":
        return any(_line_invokes_compute(arg, depth=depth + 1) for arg in words[1:])
    payload = _shell_interpreter_payload(words)
    return payload is not None and _line_invokes_compute(payload, depth=depth + 1)


def _line_invokes_compute(line: str, *, depth: int = 0) -> bool:
    return any(
        _segment_invokes_compute(segment, depth=depth)
        for segment in _split_shell_segments(line)
    )


def _is_compute_paid_command(command: str) -> bool:
    heredoc_end: str | None = None
    for raw_line in command.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if heredoc_end is not None:
            if line == heredoc_end:
                heredoc_end = None
            continue
        if re.search(r"\$\(\(", line) or re.search(
            r"(?:^|[;&|]\s*)(?:for\s*)?\(\(", line
        ):
            return True
        if _line_invokes_compute(line):
            return True
        match = re.search(r"<<-?\s*(['\"]?)([A-Za-z0-9_./-]+)\1", line)
        heredoc_end = match.group(2) if match else None
    return False


def _is_free_environment(
    command: str, tokens: tuple[str, ...]
) -> bool:
    if _SHELL_CONTROL.search(command) or not tokens:
        return False
    head, args = tokens[0], list(tokens[1:])
    if head == "pwd":
        return not args or args in (["-P"], ["-L"])
    if head == "ls":
        for arg in args:
            if arg.startswith("-"):
                if not all(char.isalnum() or char in "-_.,=" for char in arg):
                    return False
                continue
            if ".." in PurePosixPath(arg).parts or any(char in arg for char in "*?[]"):
                return False
            if not (
                arg in {".", "./", "logs", "/app", "/logs"}
                or arg.startswith(("./", "logs/", "/app/", "/logs/"))
            ):
                return False
        return True
    if head in {"which", "type"}:
        return bool(args) and all(
            "/" not in arg
            and ".." not in PurePosixPath(arg).parts
            and not any(char in arg for char in "*?[]")
            and bool(arg)
            for arg in args
        )
    if head == "cat":
        if not args:
            return False
        return all(
            not arg.startswith("-")
            and ".." not in PurePosixPath(arg).parts
            and not any(char in arg for char in "*?[]")
            and (
                arg
                in {
                    "/logs/artifacts/answer.txt",
                    "logs/artifacts/answer.txt",
                    "./logs/artifacts/answer.txt",
                }
                or arg.startswith(("/logs/agent/", "logs/agent/", "./logs/agent/"))
            )
            for arg in args
        )
    return False


class ComputeToolsPolicy(ActionAccountingPolicy):
    """Match the formal cross-domain ``compute_tools`` accounting policy."""

    policy_name = "compute_tools"

    def classify_action(self, command: str) -> ActionClass:
        stripped = command.strip()
        if not stripped:
            return ActionClass.INVALID
        if _BLOCKED_TERMS.search(stripped):
            return ActionClass.BLOCKED

        tokens = _tokens(stripped)
        if not tokens:
            return ActionClass.INVALID
        head = tokens[0]

        if head == "focus_problem" and len(tokens) == 2:
            return ActionClass.FREE_BOOKKEEPING
        if head == "shelve_problem" and len(tokens) == 1:
            return ActionClass.FREE_BOOKKEEPING
        if head in {"contest_status", "remaining_budget", "status"} and len(tokens) == 1:
            return ActionClass.FREE_STATUS
        if head == "mark_task_complete" and len(tokens) == 1:
            return ActionClass.FREE_COMPLETION
        if _is_free_environment(stripped, tokens):
            return ActionClass.FREE_ENVIRONMENT
        if _is_compute_paid_command(stripped):
            return ActionClass.COUNTED
        compound = _compound_passive_class(stripped)
        if compound is not None:
            return compound
        if _is_free_runtime_state(stripped, tokens):
            return ActionClass.FREE_ENVIRONMENT
        heredoc_target = _pure_heredoc_target(stripped)
        if heredoc_target is not None and _is_safe_local_path(
            heredoc_target, mutating=True
        ):
            return (
                ActionClass.FREE_FINALIZATION
                if _is_final_artifact(heredoc_target)
                else ActionClass.FREE_FILE_WRITE
            )
        if _is_dedicated_final_write(tokens) or _is_simple_final_redirection(
            stripped, tokens
        ):
            return ActionClass.FREE_FINALIZATION
        if _is_simple_file_write(stripped, tokens):
            return ActionClass.FREE_FILE_WRITE
        if _is_passive_file_read(stripped, tokens):
            return ActionClass.FREE_ENVIRONMENT
        mutation_target = _passive_file_mutation_target(stripped, tokens)
        if mutation_target is not None:
            return (
                ActionClass.FREE_FINALIZATION
                if tokens[0] in {"cp", "mv", "touch", "truncate"}
                and _is_final_artifact(mutation_target)
                else ActionClass.FREE_FILE_WRITE
            )

        # The formal policy is fail-closed for accounting: only actions that
        # match one of the explicit free categories above avoid the budget.
        # This prevents unlisted build, test, debugger, or local executable
        # commands from bypassing both accounting and problem attribution.
        return ActionClass.COUNTED


class CodingAllNonfreePolicy(ActionAccountingPolicy):
    """Retain the legacy non-paper Coding policy for old replay inspection."""

    policy_name = "all_nonfree"

    def classify_action(self, command: str) -> ActionClass:
        stripped = command.strip()
        if not stripped:
            return ActionClass.INVALID
        if _BLOCKED_TERMS.search(stripped):
            return ActionClass.BLOCKED

        tokens = _tokens(stripped)
        if not tokens:
            return ActionClass.INVALID
        head = tokens[0]
        if head == "focus_problem" and len(tokens) == 2:
            return ActionClass.FREE_BOOKKEEPING
        if head == "shelve_problem":
            return ActionClass.FREE_BOOKKEEPING
        if head in {"contest_status", "remaining_budget", "status"} and len(tokens) == 1:
            return ActionClass.FREE_STATUS
        if head == "mark_task_complete" and len(tokens) == 1:
            return ActionClass.FREE_COMPLETION
        if _is_simple_free_environment(stripped, tokens):
            return ActionClass.FREE_ENVIRONMENT

        # Legacy Coding replays counted all other executable terminal commands.
        return ActionClass.COUNTED


_FREE_CAT_PATHS = {
    "/etc/hostname",
    "/etc/os-release",
    "/usr/lib/os-release",
}
_FREE_LS_PATHS = {
    "/",
    "/bin",
    "/usr",
    "/usr/bin",
    "/usr/local",
    "/usr/local/bin",
}


def _is_simple_free_environment(
    command: str, tokens: tuple[str, ...]
) -> bool:
    if _SHELL_CONTROL.search(command) or not tokens:
        return False
    head = tokens[0]
    if head == "pwd":
        return len(tokens) == 1
    if head in {"which", "type"}:
        return len(tokens) >= 2 and all(
            re.fullmatch(r"[A-Za-z0-9_.+-]+", token) for token in tokens[1:]
        )
    if head == "cat":
        args = list(tokens[1:])
        if args[:1] == ["--"]:
            args = args[1:]
        return len(args) == 1 and args[0] in _FREE_CAT_PATHS
    if head == "ls":
        args = [token for token in tokens[1:] if not token.startswith("-")]
        return len(args) == 1 and args[0] in _FREE_LS_PATHS
    return False


_DEFAULT_POLICY = ComputeToolsPolicy()


def policy_from_name(name: str) -> ActionAccountingPolicy:
    """Resolve a public policy name without falling back to a fake default."""

    normalized = name.strip().lower()
    if normalized == "compute_tools":
        return ComputeToolsPolicy()
    if normalized in {"all_nonfree", "coding_all_nonfree"}:
        return CodingAllNonfreePolicy()
    raise ValueError(f"unsupported action accounting policy: {name!r}")


def classify_action(
    command: str, policy: ActionAccountingPolicy | None = None
) -> ActionClass:
    """Classify one command under the selected policy without side effects."""

    return (policy or _DEFAULT_POLICY).classify_action(command)


def apply_budget_decision(
    command: str,
    budget_state: ActionBudget,
    policy: ActionAccountingPolicy | None = None,
) -> ActionDecision:
    """Apply unit-cost accounting without ever executing the command."""

    classified = classify_action(command, policy)
    before = budget_state.remaining

    if classified == ActionClass.INVALID:
        return ActionDecision(
            command=command,
            action_class=classified,
            classified_as=classified,
            allowed=False,
            counted=False,
            executed=False,
            budget_consumed=0,
            budget_before=before,
            budget_after=budget_state.remaining,
            reason="empty_or_malformed_command",
        )
    if classified == ActionClass.BLOCKED:
        budget_state.record_blocked()
        return ActionDecision(
            command=command,
            action_class=classified,
            classified_as=classified,
            allowed=False,
            counted=False,
            executed=False,
            budget_consumed=0,
            budget_before=before,
            budget_after=budget_state.remaining,
            reason="command_blocked_by_policy",
        )
    if classified == ActionClass.COUNTED:
        if not budget_state.consume():
            return ActionDecision(
                command=command,
                action_class=ActionClass.BLOCKED,
                classified_as=classified,
                allowed=False,
                counted=True,
                executed=False,
                budget_consumed=0,
                budget_before=before,
                budget_after=budget_state.remaining,
                reason="counted_action_budget_exhausted",
            )
        return ActionDecision(
            command=command,
            action_class=classified,
            classified_as=classified,
            allowed=True,
            counted=True,
            executed=True,
            budget_consumed=1,
            budget_before=before,
            budget_after=budget_state.remaining,
            reason="counted_action_accepted",
        )

    reason = (
        "audit_only_action_accepted"
        if classified == ActionClass.AUDIT_ONLY
        else "free_action_accepted"
    )
    return ActionDecision(
        command=command,
        action_class=classified,
        classified_as=classified,
        allowed=True,
        counted=False,
        executed=True,
        budget_consumed=0,
        budget_before=before,
        budget_after=budget_state.remaining,
        reason=reason,
    )


__all__ = [
    "ActionAccountingPolicy",
    "ActionClass",
    "ActionDecision",
    "CodingAllNonfreePolicy",
    "ComputeToolsPolicy",
    "apply_budget_decision",
    "classify_action",
    "policy_from_name",
]
