"""Gated Harbor + Docker + Terminus-2 backend contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from r3bench.agentic.action_accounting import (
    ActionClass,
    apply_budget_decision,
    classify_action,
)
from r3bench.agentic.budget import ActionBudget
from r3bench.agentic.protocol_contract import (
    has_exact_paper_sandbox_limits,
    paper_sandbox_limits,
)
from r3bench.agentic.scope import AgenticScopeState
from r3bench.agentic.task_export import (
    AgenticTaskExportError,
    compute_task_fingerprint,
)
from r3bench.common.config import load_config
from r3bench.common.profile_registry import (
    DEFAULT_MODEL_PROFILES_PATH,
    ModelProfile,
    load_model_profiles,
)
from r3bench.resource_paths import resolve_path


_COMMAND = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENV = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_FILES = {
    "backend_summary.json",
    "public_action_log.json",
    "final_artifacts_manifest.json",
    "trajectory.json",
}
_TASK_FILES = {
    "budget_config.json",
    "expected_artifacts.json",
    "instruction.md",
    "public_problem_manifest.json",
    "task_config.json",
}
_ATIF_VERSION = re.compile(r"^ATIF-v1\.[0-7]$")
_CREDENTIAL_VALUE = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{16,}\b|\bhf_[A-Za-z0-9]{16,}\b|"
    r"\bolp_[A-Za-z0-9]{16,}\b|Bearer\s+[A-Za-z0-9._~+/-]{16,})"
)
_FORBIDDEN_TRAJECTORY_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "provider_headers",
        "provider_request_id",
        "request_headers",
    }
)
_RETURNED_ARTIFACT_FIELDS = frozenset(
    {
        "container_path",
        "problem_label",
        "required",
        "exists",
        "size_bytes",
        "sha256",
        "artifact_relative_path",
    }
)
_FINAL_REDIRECT = re.compile(
    r"(?:^|[ \t])(?P<operator>>{1,2})[ \t]*(?P<target>\S+)[ \t]*$"
)
_SHELL_HEREDOC_HEADERS = (
    re.compile(
        r"^cat\s*>\s*(?P<target>\S+)\s+"
        r"(?P<operator><<-?)\s*(?P<quote>['\"]?)"
        r"(?P<delimiter>[A-Za-z0-9_./-]+)(?P=quote)\s*$"
    ),
    re.compile(
        r"^cat\s+(?P<operator><<-?)\s*(?P<quote>['\"]?)"
        r"(?P<delimiter>[A-Za-z0-9_./-]+)(?P=quote)\s*"
        r">\s*(?P<target>\S+)\s*$"
    ),
    re.compile(
        r"^tee\s+(?P<target>\S+)\s+"
        r"(?P<operator><<-?)\s*(?P<quote>['\"]?)"
        r"(?P<delimiter>[A-Za-z0-9_./-]+)(?P=quote)\s*$"
    ),
)


class ExternalAgenticBackendError(RuntimeError):
    """Raised when an external runtime is absent or violates its handoff."""


@dataclass(frozen=True, slots=True)
class ExternalAgenticBackendConfig:
    status: str
    protocol_version: str
    backend: str
    executable: str | None
    harbor_executable: str
    docker_executable: str
    credential_env: str | None
    timeout_seconds: int
    trajectory_format: str
    domain_sandbox_limits: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True, slots=True)
class AgenticExecutionProfile:
    """Immutable model/API parameters passed to and attested by Harbor."""

    model_key: str
    public_model_id: str
    thinking_enabled: bool
    reasoning_effort: str | None
    temperature: float | None
    top_p: float | None

    @staticmethod
    def _parameter(value: str | float | None) -> dict[str, Any]:
        if value is None:
            return {"state": "omitted"}
        return {"state": "value", "value": value}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "model_key": self.model_key,
            "public_model_id": self.public_model_id,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self._parameter(self.reasoning_effort),
            "temperature": self._parameter(self.temperature),
            "top_p": self._parameter(self.top_p),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def resolve_agentic_execution_profile(
    model_key: str,
    model_profiles_path: str | Path = DEFAULT_MODEL_PROFILES_PATH,
) -> AgenticExecutionProfile:
    """Resolve one released ModelProfile into the external execution contract."""

    try:
        model: ModelProfile = load_model_profiles(
            resolve_path(model_profiles_path)
        )[model_key]
    except (KeyError, OSError, ValueError) as exc:
        raise ExternalAgenticBackendError(
            "real Agentic execution requires a valid released model profile"
        ) from exc
    if model.status != "release" or model.requires_owner_approval:
        raise ExternalAgenticBackendError(
            "real Agentic execution requires a released model profile"
        )
    if not isinstance(model.thinking_enabled, bool):
        raise ExternalAgenticBackendError(
            "real Agentic thinking_enabled must be resolved"
        )
    if model.reasoning_effort == "unresolved":
        raise ExternalAgenticBackendError(
            "real Agentic reasoning_effort must be resolved or omitted"
        )
    if model.temperature == "unresolved" or model.top_p == "unresolved":
        raise ExternalAgenticBackendError(
            "real Agentic sampling parameters must be resolved or omitted"
        )
    if model.reasoning_effort is not None and not isinstance(
        model.reasoning_effort, str
    ):
        raise ExternalAgenticBackendError(
            "real Agentic reasoning_effort is invalid"
        )
    for value, field in (
        (model.temperature, "temperature"),
        (model.top_p, "top_p"),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ExternalAgenticBackendError(
                f"real Agentic {field} is invalid"
            )
    return AgenticExecutionProfile(
        model_key=model.model_key,
        public_model_id=model.public_model_id,
        thinking_enabled=model.thinking_enabled,
        reasoning_effort=model.reasoning_effort,
        temperature=(
            float(model.temperature) if model.temperature is not None else None
        ),
        top_p=float(model.top_p) if model.top_p is not None else None,
    )


def _released_execution_profile(
    profile: AgenticExecutionProfile,
) -> AgenticExecutionProfile:
    released = resolve_agentic_execution_profile(profile.model_key)
    if profile != released:
        raise ExternalAgenticBackendError(
            "external execution profile differs from the released model profile"
        )
    return released


@dataclass(frozen=True, slots=True)
class _ATIFAction:
    function_name: str
    command: str
    final_path: str | None = None
    final_content_sha256: str | None = None


def _shell_heredoc_header(command: str) -> re.Match[str] | None:
    lines = command.splitlines()
    if not lines:
        return None
    return next(
        (
            match
            for pattern in _SHELL_HEREDOC_HEADERS
            if (match := pattern.fullmatch(lines[0].strip())) is not None
        ),
        None,
    )


def _shell_final_target(command: str) -> str | None:
    heredoc = _shell_heredoc_header(command)
    if heredoc is not None:
        return heredoc.group("target")
    if "\n" not in command and "\r" not in command:
        redirect = _FINAL_REDIRECT.search(command)
        if redirect is not None:
            return redirect.group("target")
    try:
        tokens = tuple(shlex.split(command, posix=True))
    except ValueError:
        return None
    if (
        len(tokens) == 2
        and tokens[0] == "write_final_artifact"
    ):
        return tokens[1]
    if tokens and tokens[0] in {"cp", "mv", "touch", "truncate"}:
        paths = tuple(token for token in tokens[1:] if not token.startswith("-"))
        return paths[-1] if paths else None
    return None


def _artifact_aliases(container_path: str) -> frozenset[str]:
    aliases = {container_path}
    if container_path.startswith("/app/"):
        relative = container_path.removeprefix("/app/")
        aliases.update({relative, f"artifacts/{relative}"})
    elif container_path == "/logs/artifacts/answer.txt":
        aliases.add("logs/artifacts/answer.txt")
    return frozenset(aliases)


def _shell_final_path(
    command: str, designated_artifacts: frozenset[str]
) -> str | None:
    if classify_action(command) != ActionClass.FREE_FINALIZATION:
        return None
    target = _shell_final_target(command)
    if target is None:
        return None
    normalized = target.strip("'\"")
    if normalized.startswith("./"):
        normalized = normalized.removeprefix("./")
    matched: list[str] = []
    for container_path in sorted(designated_artifacts):
        if normalized in _artifact_aliases(container_path):
            matched.append(container_path)
    if len(matched) != 1:
        raise ExternalAgenticBackendError(
            "external backend shell finalization does not bind one artifact"
        )
    return matched[0]


def _printf_literal_bytes(format_string: str, arguments: tuple[str, ...]) -> bytes | None:
    output = bytearray()
    argument_index = 0
    escapes = {
        "a": b"\a",
        "b": b"\b",
        "f": b"\f",
        "n": b"\n",
        "r": b"\r",
        "t": b"\t",
        "v": b"\v",
        "\\": b"\\",
    }
    index = 0
    while index < len(format_string):
        character = format_string[index]
        if character == "\\":
            index += 1
            if index >= len(format_string) or format_string[index] not in escapes:
                return None
            output.extend(escapes[format_string[index]])
        elif character == "%":
            index += 1
            if index >= len(format_string):
                return None
            conversion = format_string[index]
            if conversion == "%":
                output.extend(b"%")
            elif conversion == "s" and argument_index < len(arguments):
                value = arguments[argument_index]
                if "\x00" in value:
                    return None
                output.extend(value.encode("utf-8"))
                argument_index += 1
            else:
                return None
        else:
            if character == "\x00":
                return None
            output.extend(character.encode("utf-8"))
        index += 1
    if argument_index != len(arguments):
        return None
    return bytes(output)


def _shell_heredoc_bytes(command: str) -> bytes | None:
    if "\r" in command:
        return None
    lines = command.splitlines()
    header = _shell_heredoc_header(command)
    if header is None or not header.group("quote"):
        return None
    delimiter = header.group("delimiter")
    strip_tabs = header.group("operator") == "<<-"
    closing: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        candidate = line.lstrip("\t") if strip_tabs else line
        if candidate == delimiter:
            closing = index
            break
    if closing is None or any(line.strip() for line in lines[closing + 1 :]):
        return None
    body = lines[1:closing]
    if strip_tabs:
        body = [line.lstrip("\t") for line in body]
    text = "\n".join(body) + ("\n" if body else "")
    return text.encode("utf-8")


def _shell_final_content_sha256(command: str) -> str | None:
    heredoc = _shell_heredoc_bytes(command)
    if heredoc is not None:
        return hashlib.sha256(heredoc).hexdigest()
    if "\n" in command or "\r" in command:
        return None
    redirect = _FINAL_REDIRECT.search(command)
    if redirect is None or redirect.group("operator") != ">":
        return None
    writer = command[: redirect.start()].strip()
    try:
        tokens = tuple(shlex.split(writer, posix=True))
    except ValueError:
        return None
    content: bytes | None = None
    if tokens and tokens[0] == "echo":
        arguments = tokens[1:]
        if not (
            arguments
            and arguments[0].startswith("-")
            or any("\\" in argument or "\x00" in argument for argument in arguments)
        ):
            content = (" ".join(arguments) + "\n").encode("utf-8")
    elif len(tokens) >= 2 and tokens[0] == "printf":
        content = _printf_literal_bytes(tokens[1], tokens[2:])
    return hashlib.sha256(content).hexdigest() if content is not None else None


def load_external_backend_config(path: str | Path) -> ExternalAgenticBackendConfig:
    value = load_config(path)
    expected = {
        "schema_version",
        "status",
        "protocol_version",
        "backend",
        "executable",
        "harbor_executable",
        "docker_executable",
        "credential_env",
        "timeout_seconds",
        "trajectory_format",
        "domain_sandbox_limits",
    }
    if set(value) != expected or value.get("schema_version") != "2.0":
        raise ExternalAgenticBackendError("external backend config fields are invalid")
    status = value.get("status")
    if status not in {"not_configured", "configured"}:
        raise ExternalAgenticBackendError("external backend status is invalid")
    executable = value.get("executable")
    backend = value.get("backend")
    if backend != "harbor_terminus2":
        raise ExternalAgenticBackendError(
            "external backend must be harbor_terminus2"
        )
    protocol_version = value.get("protocol_version")
    if not isinstance(protocol_version, str) or not protocol_version.strip():
        raise ExternalAgenticBackendError("protocol_version must be non-empty")
    credential_env = value.get("credential_env")
    if executable is not None and (
        not isinstance(executable, str) or not _COMMAND.fullmatch(executable)
    ):
        raise ExternalAgenticBackendError(
            "external backend executable must be a command name, not a path"
        )
    if credential_env is not None and (
        not isinstance(credential_env, str) or not _ENV.fullmatch(credential_env)
    ):
        raise ExternalAgenticBackendError("credential_env must name an environment variable")
    harbor_executable = value.get("harbor_executable")
    docker_executable = value.get("docker_executable")
    if not isinstance(harbor_executable, str) or not _COMMAND.fullmatch(
        harbor_executable
    ):
        raise ExternalAgenticBackendError("harbor_executable must be a command name")
    if not isinstance(docker_executable, str) or not _COMMAND.fullmatch(
        docker_executable
    ):
        raise ExternalAgenticBackendError("docker_executable must be a command name")
    trajectory_format = value.get("trajectory_format")
    if trajectory_format != "ATIF":
        raise ExternalAgenticBackendError("trajectory_format must be ATIF")
    domain_sandbox_limits = value.get("domain_sandbox_limits")
    if not has_exact_paper_sandbox_limits(domain_sandbox_limits):
        raise ExternalAgenticBackendError(
            "domain_sandbox_limits differ from the paper protocol"
        )
    timeout = value.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ExternalAgenticBackendError("timeout_seconds must be positive")
    if status == "configured" and (executable is None or credential_env is None):
        raise ExternalAgenticBackendError(
            "configured external backend requires executable and credential_env"
        )
    return ExternalAgenticBackendConfig(
        status=status,
        protocol_version=protocol_version,
        backend=backend,
        executable=executable,
        harbor_executable=harbor_executable,
        docker_executable=docker_executable,
        credential_env=credential_env,
        timeout_seconds=timeout,
        trajectory_format=trajectory_format,
        domain_sandbox_limits={
            str(domain): dict(limits)
            for domain, limits in domain_sandbox_limits.items()
        },
    )


def _capabilities_compatible(
    capabilities: object, config: ExternalAgenticBackendConfig
) -> bool:
    return bool(
        isinstance(capabilities, dict)
        and capabilities.get("r3bench_agentic_protocol")
        == config.protocol_version
        and capabilities.get("backend") == "harbor"
        and capabilities.get("environment") == "docker"
        and capabilities.get("agent") == "terminus-2"
        and capabilities.get("action_policy") == "compute_tools"
        and capabilities.get("os_command_execution_available") is True
        and capabilities.get("supports_compilation_and_tests") is True
        and capabilities.get("writes_complete_trajectory") is True
        and capabilities.get("trajectory_format") == config.trajectory_format
        and capabilities.get("enforces_domain_sandbox_limits") is True
    )


def check_external_backend_readiness(
    config: ExternalAgenticBackendConfig,
    *,
    probe: bool = False,
) -> dict[str, Any]:
    executable_found = bool(config.executable and shutil.which(config.executable))
    harbor_found = bool(shutil.which(config.harbor_executable))
    docker_found = bool(shutil.which(config.docker_executable))
    credential_available = bool(
        config.credential_env and os.environ.get(config.credential_env)
    )
    protocol_compatible = False
    process_started = False
    if probe and config.status == "configured" and executable_found:
        process_started = True
        try:
            completed = subprocess.run(
                [str(config.executable), "capabilities", "--json"],
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=min(config.timeout_seconds, 30),
                env=os.environ.copy(),
            )
            capabilities = json.loads(completed.stdout) if completed.returncode == 0 else {}
            protocol_compatible = _capabilities_compatible(capabilities, config)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            protocol_compatible = False
    status = (
        "not_configured"
        if config.status != "configured"
        else "not_ready"
        if not executable_found
        or not harbor_found
        or not docker_found
        or not credential_available
        else "ready"
        if not probe or protocol_compatible
        else "protocol_incompatible"
    )
    return {
        "schema_version": "2.0",
        "status": status,
        "protocol_version": config.protocol_version,
        "backend": config.backend,
        "executable_found": executable_found,
        "harbor_executable_found": harbor_found,
        "docker_executable_found": docker_found,
        "agent": "terminus-2",
        "environment": "docker",
        "domain_sandbox_limits": {
            domain: dict(limits)
            for domain, limits in config.domain_sandbox_limits.items()
        },
        "credential_available": credential_available,
        "protocol_compatible": protocol_compatible if probe else None,
        "probe_requested": probe,
        "external_process_started": process_started,
        "credential_value_serialized": False,
        "private_config_serialized": False,
    }


def _object(path: Path, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalAgenticBackendError(f"external backend {kind} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ExternalAgenticBackendError(f"external backend {kind} must be an object")
    return value


def _exact_attested_value(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, int)
            and actual == expected
        )
    if expected is None:
        return actual is None
    return actual == expected


def _verified_task_fingerprint(task_dir: Path) -> str:
    if not task_dir.is_dir():
        raise ExternalAgenticBackendError("task_dir is not an exported task directory")
    children = {child.name for child in task_dir.iterdir()}
    if children != _TASK_FILES or any(
        not (task_dir / name).is_file() or (task_dir / name).is_symlink()
        for name in _TASK_FILES
    ):
        raise ExternalAgenticBackendError(
            "task_dir does not contain exactly one regular exported task contract"
        )
    task_config = _object(task_dir / "task_config.json", "task config")
    recorded = task_config.get("task_fingerprint_sha256")
    if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
        raise ExternalAgenticBackendError(
            "exported task has no valid task fingerprint"
        )
    try:
        actual = compute_task_fingerprint(task_dir)
    except (AgenticTaskExportError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalAgenticBackendError(
            "exported task fingerprint could not be recomputed"
        ) from exc
    if actual != recorded:
        raise ExternalAgenticBackendError(
            "exported task differs from its recorded fingerprint"
        )
    return recorded


def _contains_forbidden_trajectory_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in _FORBIDDEN_TRAJECTORY_KEYS
            or _contains_forbidden_trajectory_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_trajectory_key(item) for item in value)
    return False


def _atif_content(value: object) -> bool:
    if isinstance(value, str):
        return True
    return bool(
        isinstance(value, list)
        and value
        and all(isinstance(part, dict) for part in value)
    )


def _atif_timestamp(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str) or "T" not in value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _validate_atif_step(
    step: object,
    expected_id: int,
    expected_execution_profile: AgenticExecutionProfile,
) -> int:
    if not isinstance(step, dict) or step.get("step_id") != expected_id:
        raise ExternalAgenticBackendError(
            "external backend trajectory step IDs are not contiguous"
        )
    source = step.get("source")
    if source not in {"system", "user", "agent"}:
        raise ExternalAgenticBackendError(
            "external backend trajectory has an invalid step source"
        )
    if "message" not in step or not _atif_content(step["message"]):
        raise ExternalAgenticBackendError(
            "external backend trajectory step message is missing or invalid"
        )
    if not _atif_timestamp(step.get("timestamp")):
        raise ExternalAgenticBackendError(
            "external backend trajectory step timestamp is not ISO 8601"
        )
    agent_only = {
        "llm_call_count",
        "model_name",
        "reasoning_effort",
        "reasoning_content",
        "tool_calls",
        "metrics",
    }
    if source != "agent" and any(field in step for field in agent_only):
        raise ExternalAgenticBackendError(
            "external backend trajectory uses agent-only fields on another source"
        )
    llm_calls = step.get("llm_call_count")
    if source == "agent" and (
        isinstance(llm_calls, bool) or not isinstance(llm_calls, int) or llm_calls < 0
    ):
        raise ExternalAgenticBackendError(
            "external backend agent step has no valid llm_call_count"
        )
    if llm_calls == 0 and any(
        field in step
        for field in (
            "metrics",
            "model_name",
            "reasoning_content",
            "reasoning_effort",
        )
    ):
        raise ExternalAgenticBackendError(
            "external backend deterministic step contains LLM-only data"
        )
    if isinstance(llm_calls, int) and llm_calls > 0:
        expected_effort = expected_execution_profile.reasoning_effort
        effort_matches = (
            "reasoning_effort" not in step
            if expected_effort is None
            else step.get("reasoning_effort") == expected_effort
        )
        if (
            step.get("model_name") != expected_execution_profile.public_model_id
            or not effort_matches
        ):
            raise ExternalAgenticBackendError(
                "external backend LLM step differs from the execution profile"
            )

    calls = step.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ExternalAgenticBackendError(
            "external backend trajectory tool_calls must be an array"
        )
    call_ids: set[str] = set()
    for call in calls:
        if not isinstance(call, dict):
            raise ExternalAgenticBackendError(
                "external backend trajectory has a malformed tool call"
            )
        call_id = call.get("tool_call_id")
        function_name = call.get("function_name")
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id in call_ids
            or not isinstance(function_name, str)
            or not function_name
            or not isinstance(call.get("arguments"), dict)
        ):
            raise ExternalAgenticBackendError(
                "external backend trajectory has a malformed tool call"
            )
        call_ids.add(call_id)

    observation = step.get("observation")
    if observation is None:
        return int(llm_calls or 0)
    if source == "user" or not isinstance(observation, dict):
        raise ExternalAgenticBackendError(
            "external backend trajectory has a malformed observation"
        )
    results = observation.get("results")
    if not isinstance(results, list):
        raise ExternalAgenticBackendError(
            "external backend trajectory observation results must be an array"
        )
    for result in results:
        if not isinstance(result, dict):
            raise ExternalAgenticBackendError(
                "external backend trajectory has a malformed observation result"
            )
        source_call_id = result.get("source_call_id")
        if source_call_id is not None and source_call_id not in call_ids:
            raise ExternalAgenticBackendError(
                "external backend trajectory observation references an unknown tool call"
            )
        if "content" in result and not _atif_content(result["content"]):
            raise ExternalAgenticBackendError(
                "external backend trajectory observation content is invalid"
            )
    return int(llm_calls or 0)


def _normalized_atif_action(
    call: Mapping[str, Any], *, designated_artifacts: frozenset[str]
) -> _ATIFAction:
    function_name = str(call["function_name"])
    arguments = call["arguments"]
    if function_name in {"bash_command", "shell", "terminal"}:
        if set(arguments) != {"command"}:
            raise ExternalAgenticBackendError(
                "external backend shell tool arguments are not normalized"
            )
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ExternalAgenticBackendError(
                "external backend shell tool has no command"
            )
        final_path = _shell_final_path(command, designated_artifacts)
        return _ATIFAction(
            function_name=function_name,
            command=command,
            final_path=final_path,
            final_content_sha256=(
                _shell_final_content_sha256(command)
                if final_path is not None
                else None
            ),
        )
    if function_name == "focus_problem":
        if set(arguments) != {"problem_id"} or not isinstance(
            arguments.get("problem_id"), str
        ):
            raise ExternalAgenticBackendError(
                "external backend focus_problem arguments are invalid"
            )
        return _ATIFAction(
            function_name=function_name,
            command=f"focus_problem {arguments['problem_id']}",
        )
    if function_name in {
        "shelve_problem",
        "contest_status",
        "remaining_budget",
        "mark_task_complete",
    }:
        if arguments:
            raise ExternalAgenticBackendError(
                f"external backend {function_name} arguments are invalid"
            )
        return _ATIFAction(function_name=function_name, command=function_name)
    if function_name == "write_final":
        if set(arguments) != {"path", "content"}:
            raise ExternalAgenticBackendError(
                "external backend write_final arguments are invalid"
            )
        final_path = arguments.get("path")
        content = arguments.get("content")
        if (
            not isinstance(final_path, str)
            or final_path not in designated_artifacts
            or not isinstance(content, str)
        ):
            raise ExternalAgenticBackendError(
                "external backend write_final differs from the artifact contract"
            )
        return _ATIFAction(
            function_name=function_name,
            command=f"write_final_artifact {final_path}",
            final_path=final_path,
            final_content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
    raise ExternalAgenticBackendError(
        f"external backend used unsupported native tool {function_name!r}"
    )


def _validate_atif_trajectory(
    path: Path,
    *,
    expected_model_key: str | None,
    expected_execution_profile: AgenticExecutionProfile,
    designated_artifacts: frozenset[str],
) -> tuple[str, dict[tuple[int, str], _ATIFAction], int]:
    trajectory = _object(path, "ATIF trajectory")
    if not _ATIF_VERSION.fullmatch(str(trajectory.get("schema_version", ""))):
        raise ExternalAgenticBackendError(
            "external backend trajectory is not a supported ATIF version"
        )
    agent = trajectory.get("agent")
    if not isinstance(agent, dict) or agent.get("name") != "terminus-2":
        raise ExternalAgenticBackendError(
            "external backend trajectory was not produced by Terminus-2"
        )
    if not isinstance(agent.get("version"), str) or not agent["version"]:
        raise ExternalAgenticBackendError(
            "external backend trajectory has no AgentSchema agent version"
        )
    if (
        expected_model_key is not None
        and expected_model_key != expected_execution_profile.model_key
    ):
        raise ExternalAgenticBackendError(
            "external execution profile and requested model disagree"
        )
    if (
        agent.get("model_name") != expected_execution_profile.public_model_id
        or agent.get("execution_profile")
        != expected_execution_profile.to_dict()
    ):
        raise ExternalAgenticBackendError(
            "external backend ATIF execution profile differs from the requested "
            "model profile"
        )
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ExternalAgenticBackendError(
            "external backend trajectory has no interaction steps"
        )
    llm_call_count = sum(
        _validate_atif_step(step, expected_id, expected_execution_profile)
        for expected_id, step in enumerate(steps, start=1)
    )
    if llm_call_count <= 0:
        raise ExternalAgenticBackendError(
            "external backend trajectory has no recorded model API call"
        )
    if _contains_forbidden_trajectory_key(trajectory) or _CREDENTIAL_VALUE.search(
        path.read_text(encoding="utf-8")
    ):
        raise ExternalAgenticBackendError(
            "external backend trajectory contains credential-bearing metadata"
        )
    commands: dict[tuple[int, str], _ATIFAction] = {}
    for step in steps:
        step_id = int(step["step_id"])
        for call in step.get("tool_calls", []):
            commands[(step_id, str(call["tool_call_id"]))] = _normalized_atif_action(
                call, designated_artifacts=designated_artifacts
            )
    return hashlib.sha256(path.read_bytes()).hexdigest(), commands, llm_call_count


def _validate_public_action_log(
    action: Mapping[str, Any],
    commands: Mapping[tuple[int, str], _ATIFAction],
    *,
    task_budget: object,
    problem_labels: Mapping[str, str],
) -> tuple[int, int, tuple[tuple[int, str], ...]]:
    if action.get("policy") != "compute_tools":
        raise ExternalAgenticBackendError(
            "external backend action policy differs from the paper contract"
        )
    if action.get("budget") != task_budget:
        raise ExternalAgenticBackendError(
            "external backend budget differs from the task"
        )
    rows = action.get("actions")
    if not isinstance(rows, list):
        raise ExternalAgenticBackendError(
            "external backend public action log has no actions array"
        )
    if task_budget is not None and (
        isinstance(task_budget, bool)
        or not isinstance(task_budget, int)
        or task_budget < 0
    ):
        raise ExternalAgenticBackendError(
            "external backend task has an invalid action budget"
        )
    command_order = tuple(commands)
    if len(rows) != len(command_order):
        raise ExternalAgenticBackendError(
            "external backend action log does not cover every ATIF tool call"
        )
    budget_state = ActionBudget(task_budget)
    executed_count = 0
    executed_final_writes: list[tuple[int, str]] = []
    scope = AgenticScopeState(
        valid_problem_ids=frozenset(problem_labels.values()),
        problem_labels=problem_labels,
    )
    for sequence, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get("sequence") != sequence:
            raise ExternalAgenticBackendError(
                "external backend action sequence is malformed"
            )
        step_id = row.get("source_step_id")
        call_id = row.get("tool_call_id")
        if (
            isinstance(step_id, bool)
            or not isinstance(step_id, int)
            or not isinstance(call_id, str)
        ):
            raise ExternalAgenticBackendError(
                "external backend action has an invalid ATIF binding"
            )
        key = (step_id, call_id)
        if key != command_order[sequence - 1]:
            raise ExternalAgenticBackendError(
                "external backend action order differs from ATIF tool-call order"
            )
        binding = commands[key]
        command = row.get("command")
        if (
            command != binding.command
            or row.get("function_name") != binding.function_name
        ):
            raise ExternalAgenticBackendError(
                "external backend action command differs from its ATIF tool call"
            )
        classified = classify_action(command)
        scope_decision = scope.authorize_action(classified, command)
        if not scope_decision.allowed:
            budget_before = budget_state.remaining
            budget_state.record_blocked()
            expected_action_class = ActionClass.BLOCKED
            expected_allowed = False
            expected_counted = classified == ActionClass.COUNTED
            expected_executed = False
            expected_consumed = 0
            expected_reason = scope_decision.reason
            budget_after = budget_state.remaining
        else:
            decision = apply_budget_decision(command, budget_state)
            expected_action_class = decision.action_class
            expected_allowed = decision.allowed
            expected_counted = decision.counted
            expected_executed = decision.executed
            expected_consumed = decision.budget_consumed
            expected_reason = decision.reason
            budget_before = decision.budget_before
            budget_after = decision.budget_after

        expected_core = {
            "classified_as": classified.value,
            "action_class": expected_action_class.value,
            "allowed": expected_allowed,
            "counted": expected_counted,
            "executed": expected_executed,
            "budget_consumed": expected_consumed,
            "budget_before": budget_before,
            "budget_after": budget_after,
            "reason": expected_reason,
        }
        if any(
            field not in row
            or not _exact_attested_value(row[field], expected)
            for field, expected in expected_core.items()
        ):
            raise ExternalAgenticBackendError(
                "external backend action decision differs from compute_tools replay, "
                "problem scope, or budget"
            )
        if expected_allowed and classified == ActionClass.FREE_BOOKKEEPING:
            try:
                tokens = tuple(shlex.split(command, posix=True))
                if tokens[0] == "focus_problem":
                    scope.focus_problem(tokens[1])
                elif tokens[0] == "shelve_problem":
                    scope.shelve_problem()
            except (IndexError, ValueError):
                raise ExternalAgenticBackendError(
                    "external backend action has invalid problem bookkeeping"
                ) from None
        expected_attribution = (
            scope_decision.attributed_problem_id
            if expected_counted and expected_allowed
            else None
        )
        if (
            row.get("active_problem_id") != scope.active_problem_id
            or row.get("attributed_problem_id") != expected_attribution
        ):
            raise ExternalAgenticBackendError(
                "external backend action problem scope or attribution differs"
            )
        executed_count += int(expected_executed)
        if expected_executed and binding.final_path is not None:
            executed_final_writes.append(key)
    if action.get("used") != budget_state.used:
        raise ExternalAgenticBackendError(
            "external backend action usage does not match recomputed cost"
        )
    for field, expected in (
        ("remaining", budget_state.remaining),
        ("blocked_attempts", budget_state.blocked_attempts),
        ("action_attempts", len(rows)),
    ):
        if field in action and action[field] != expected:
            raise ExternalAgenticBackendError(
                "external backend action summary differs from replay"
            )
    return budget_state.used, executed_count, tuple(executed_final_writes)


def _validate_artifact_manifest(
    scratch: Path,
    manifest: Mapping[str, Any],
    expected_artifacts: Mapping[str, Any],
) -> tuple[list[object], frozenset[str]]:
    if set(manifest) != {
        "schema_version",
        "grade_after_episode",
        "correctness_feedback_exposed",
        "artifacts",
    } or any(
        (
            manifest.get("schema_version") != "1.0",
            manifest.get("grade_after_episode") is not True,
            manifest.get("correctness_feedback_exposed") is not False,
        )
    ):
        raise ExternalAgenticBackendError(
            "external artifact manifest metadata is malformed"
        )
    rows = manifest.get("artifacts")
    expected_rows = expected_artifacts.get("artifacts")
    if not isinstance(rows, list) or not isinstance(expected_rows, list):
        raise ExternalAgenticBackendError("external artifact manifest is malformed")

    expected_by_path: dict[str, Mapping[str, Any]] = {}
    expected_relatives: set[str] = set()
    for row in expected_rows:
        if not isinstance(row, Mapping):
            raise ExternalAgenticBackendError("task artifact contract is malformed")
        container_path = row.get("container_path")
        relative = row.get("sandbox_relative_path")
        required = row.get("required")
        label = row.get("problem_label")
        if (
            not isinstance(container_path, str)
            or not container_path
            or container_path in expected_by_path
            or not isinstance(relative, str)
            or not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative in expected_relatives
            or not isinstance(required, bool)
            or (label is not None and label not in tuple("ABCDEF"))
        ):
            raise ExternalAgenticBackendError("task artifact contract is malformed")
        expected_by_path[container_path] = row
        expected_relatives.add(relative)

    returned_by_path: dict[str, Mapping[str, Any]] = {}
    declared_files: set[str] = set()
    artifact_root = scratch / "artifacts"
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ExternalAgenticBackendError("external artifact directory is missing")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _RETURNED_ARTIFACT_FIELDS:
            raise ExternalAgenticBackendError("external artifact row is malformed")
        container_path = row.get("container_path")
        if (
            not isinstance(container_path, str)
            or container_path in returned_by_path
            or container_path not in expected_by_path
        ):
            raise ExternalAgenticBackendError(
                "external artifact manifest differs from the task contract"
            )
        expected = expected_by_path[container_path]
        if (
            row.get("problem_label") != expected.get("problem_label")
            or row.get("required") is not expected.get("required")
            or not isinstance(row.get("exists"), bool)
        ):
            raise ExternalAgenticBackendError(
                "external artifact public binding differs from the task contract"
            )
        returned_by_path[container_path] = row
        exists = bool(row["exists"])
        expected_relative = (
            f"artifacts/{expected['sandbox_relative_path']}" if exists else None
        )
        size_bytes = row.get("size_bytes")
        if not exists:
            if (
                expected.get("required") is True
                or row.get("artifact_relative_path") is not None
                or row.get("sha256") is not None
                or size_bytes != 0
            ):
                raise ExternalAgenticBackendError(
                    "external missing artifact metadata is inconsistent"
                )
            continue
        if (
            row.get("artifact_relative_path") != expected_relative
            or expected_relative in declared_files
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
        ):
            raise ExternalAgenticBackendError(
                "external present artifact metadata is inconsistent"
            )
        assert isinstance(expected_relative, str)
        declared_files.add(expected_relative)
        pure = PurePosixPath(expected_relative)
        path = scratch / Path(*pure.parts)
        parents = [path]
        current = path.parent
        while current != scratch:
            parents.append(current)
            current = current.parent
        if any(candidate.is_symlink() for candidate in parents):
            raise ExternalAgenticBackendError(
                "external artifact path contains a symlink"
            )
        if not path.is_file() or path.stat().st_size != size_bytes:
            raise ExternalAgenticBackendError("external artifact file is missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256"):
            raise ExternalAgenticBackendError("external artifact digest mismatch")

    if set(returned_by_path) != set(expected_by_path):
        raise ExternalAgenticBackendError(
            "external artifact manifest differs from the task contract"
        )
    actual_files: set[str] = set()
    for candidate in artifact_root.rglob("*"):
        if candidate.is_symlink():
            raise ExternalAgenticBackendError(
                "external artifact directory contains a symlink"
            )
        if candidate.is_file():
            actual_files.add(candidate.relative_to(scratch).as_posix())
        elif not candidate.is_dir():
            raise ExternalAgenticBackendError(
                "external artifact directory contains a non-regular entry"
            )
    if actual_files != declared_files:
        raise ExternalAgenticBackendError(
            "external artifact files differ from the returned manifest"
        )
    return rows, frozenset(expected_by_path)


def _validate_final_write_bindings(
    commands: Mapping[tuple[int, str], _ATIFAction],
    executed_final_writes: tuple[tuple[int, str], ...],
    manifest_rows: list[object],
) -> None:
    latest: dict[str, str | None] = {}
    for key in executed_final_writes:
        binding = commands[key]
        if binding.final_path is not None:
            latest[binding.final_path] = binding.final_content_sha256
    manifest = {
        row.get("container_path"): row
        for row in manifest_rows
        if isinstance(row, dict)
    }
    for final_path, expected_sha256 in latest.items():
        row = manifest.get(final_path)
        if (
            not isinstance(row, dict)
            or row.get("exists") is not True
            or (
                expected_sha256 is not None
                and row.get("sha256") != expected_sha256
            )
        ):
            raise ExternalAgenticBackendError(
                "external backend write_final content differs from its artifact digest"
            )
    for final_path, row in manifest.items():
        if (
            isinstance(final_path, str)
            and isinstance(row, Mapping)
            and row.get("exists") is True
            and (
                final_path not in latest
                or latest[final_path] is None
            )
        ):
            raise ExternalAgenticBackendError(
                "external present artifact has no executed write event with "
                "verifiable content"
            )


def validate_external_backend_handoff(
    scratch: str | Path,
    task_dir: str | Path,
    *,
    expected_model_key: str | None = None,
    expected_execution_profile: AgenticExecutionProfile | None = None,
) -> dict[str, Any]:
    """Validate a real Harbor handoff and return trajectory metadata."""

    if expected_execution_profile is None:
        if expected_model_key is None:
            raise ExternalAgenticBackendError(
                "external handoff validation requires an expected model profile"
            )
        expected_execution_profile = resolve_agentic_execution_profile(
            expected_model_key
        )
    else:
        expected_execution_profile = _released_execution_profile(
            expected_execution_profile
        )
    if (
        expected_model_key is not None
        and expected_model_key != expected_execution_profile.model_key
    ):
        raise ExternalAgenticBackendError(
            "external execution profile and requested model disagree"
        )
    scratch = Path(scratch)
    task_dir = Path(task_dir)
    persisted = task_dir.absolute() == (scratch / "task_binding").absolute()
    expected_children = _ALLOWED_FILES | {"artifacts"}
    if persisted:
        expected_children |= {"saved_outputs.jsonl", "task_binding"}
    children = {child.name for child in scratch.iterdir()}
    if children != expected_children or any(
        not (scratch / name).is_file() or (scratch / name).is_symlink()
        for name in _ALLOWED_FILES
    ):
        raise ExternalAgenticBackendError(
            "external backend returned non-allowlisted files"
        )
    if persisted and (
        not (scratch / "saved_outputs.jsonl").is_file()
        or (scratch / "saved_outputs.jsonl").is_symlink()
        or not task_dir.is_dir()
        or task_dir.is_symlink()
    ):
        raise ExternalAgenticBackendError(
            "persisted external backend layout is malformed"
        )
    _verified_task_fingerprint(task_dir)
    summary = _object(scratch / "backend_summary.json", "summary")
    action = _object(scratch / "public_action_log.json", "action summary")
    manifest = _object(scratch / "final_artifacts_manifest.json", "artifact manifest")
    task = _object(task_dir / "task_config.json", "task config")
    task_budget_config = _object(task_dir / "budget_config.json", "task budget")
    expected_artifacts = _object(
        task_dir / "expected_artifacts.json", "task artifact contract"
    )
    problem_manifest = _object(
        task_dir / "public_problem_manifest.json", "public problem manifest"
    )
    domain = task.get("domain")
    try:
        expected_limits = paper_sandbox_limits(str(domain))
    except ValueError as exc:
        raise ExternalAgenticBackendError(
            "external backend task has an unsupported domain"
        ) from exc
    if (
        task.get("runtime") != "harbor_terminus2_paper_v1"
        or task.get("paper_equivalent_runtime_required") is not True
        or task.get("sandbox_limits") != expected_limits
    ):
        raise ExternalAgenticBackendError(
            "external backend task does not carry the paper runtime contract"
        )
    if summary.get("correctness_feedback_exposed") is not False:
        raise ExternalAgenticBackendError(
            "external backend exposed live correctness feedback"
        )
    required_summary = {
        "backend": "harbor",
        "environment": "docker",
        "agent": "terminus-2",
        "paper_equivalent_runtime": True,
        "os_command_execution_available": True,
        "container_runtime_called": True,
        "compilation_and_tests_available": True,
        "sandbox_limits_enforced": True,
        "model_api_called": True,
        "raw_trajectory_saved": True,
        "trajectory_complete": True,
        "trajectory_format": "ATIF",
    }
    if any(summary.get(key) != expected for key, expected in required_summary.items()):
        raise ExternalAgenticBackendError(
            "external backend summary does not attest Harbor/Terminus-2 execution"
        )
    os_commands_executed = summary.get("os_commands_executed")
    if not (
        isinstance(os_commands_executed, bool)
        or (
            isinstance(os_commands_executed, int)
            and os_commands_executed >= 0
        )
    ):
        raise ExternalAgenticBackendError(
            "external backend OS-command diagnostic is invalid"
        )
    if summary.get("task_id") != task.get("task_id"):
        raise ExternalAgenticBackendError("external backend task binding differs")
    if summary.get("suite_id") != task.get("suite_id"):
        raise ExternalAgenticBackendError("external backend suite binding differs")
    if (
        summary.get("model_key") != expected_execution_profile.model_key
        or summary.get("public_model_id")
        != expected_execution_profile.public_model_id
        or summary.get("execution_profile")
        != expected_execution_profile.to_dict()
    ):
        raise ExternalAgenticBackendError(
            "external backend summary execution profile differs from the request"
        )
    if summary.get("domain") != task.get("domain"):
        raise ExternalAgenticBackendError("external backend domain differs")
    if summary.get("sandbox_limits") != task.get("sandbox_limits"):
        raise ExternalAgenticBackendError(
            "external backend sandbox limits differ from the task contract"
        )
    if task_budget_config.get("policy") != "compute_tools":
        raise ExternalAgenticBackendError(
            "real Agentic handoff requires the cross-domain compute_tools policy"
        )
    task_budget = task_budget_config.get("counted_action_budget")
    labels = task.get("problem_labels")
    manifest_problems = problem_manifest.get("problems")
    if (
        not isinstance(labels, dict)
        or not labels
        or not all(
            isinstance(label, str)
            and isinstance(problem_id, str)
            and problem_id
            for label, problem_id in labels.items()
        )
        or not isinstance(manifest_problems, list)
    ):
        raise ExternalAgenticBackendError(
            "external backend task has an invalid problem scope contract"
        )
    manifest_labels = {
        row.get("problem_label"): row.get("problem_id")
        for row in manifest_problems
        if isinstance(row, dict)
    }
    if len(manifest_labels) != len(manifest_problems) or manifest_labels != labels:
        raise ExternalAgenticBackendError(
            "external backend task and manifest problem scopes differ"
        )
    rows, expected_paths = _validate_artifact_manifest(
        scratch, manifest, expected_artifacts
    )
    trajectory_sha256, commands, llm_call_count = _validate_atif_trajectory(
        scratch / "trajectory.json",
        expected_model_key=expected_model_key,
        expected_execution_profile=expected_execution_profile,
        designated_artifacts=frozenset(expected_paths),
    )
    if (
        isinstance(summary.get("llm_call_count"), bool)
        or summary.get("llm_call_count") != llm_call_count
    ):
        raise ExternalAgenticBackendError(
            "external backend summary LLM-call count differs from ATIF"
        )
    _, executed_count, executed_final_writes = _validate_public_action_log(
        action,
        commands,
        task_budget=task_budget,
        problem_labels={str(key): str(value) for key, value in labels.items()},
    )
    _validate_final_write_bindings(commands, executed_final_writes, rows)
    recorded_os_count = int(os_commands_executed)
    if recorded_os_count != executed_count:
        raise ExternalAgenticBackendError(
            "external backend OS-command count differs from the ATIF action log"
        )
    return {
        "trajectory_format": "ATIF",
        "trajectory_complete": True,
        "trajectory_sha256": trajectory_sha256,
    }


def run_external_agentic_backend(
    *,
    task_dir: str | Path,
    output_dir: str | Path,
    model_key: str | None = None,
    execution_profile: AgenticExecutionProfile | None = None,
    config: ExternalAgenticBackendConfig,
    allow_real_api: bool,
    allow_agentic_backend: bool,
) -> dict[str, Any]:
    """Run exactly one real Harbor/Terminus-2 task and retain its ATIF trace."""

    if execution_profile is None:
        if model_key is None:
            raise ExternalAgenticBackendError(
                "external Agentic execution requires a model execution profile"
            )
        execution_profile = resolve_agentic_execution_profile(model_key)
    else:
        execution_profile = _released_execution_profile(execution_profile)
    if model_key is not None and model_key != execution_profile.model_key:
        raise ExternalAgenticBackendError(
            "external execution profile and requested model disagree"
        )
    model_key = execution_profile.model_key
    if not allow_real_api or not allow_agentic_backend:
        raise ExternalAgenticBackendError(
            "external Agentic execution requires --allow-real-api and "
            "--allow-agentic-backend"
        )
    readiness = check_external_backend_readiness(config, probe=True)
    if readiness["status"] != "ready":
        raise ExternalAgenticBackendError("external Agentic backend is not ready")
    task = Path(task_dir)
    target = Path(output_dir)
    task_fingerprint = _verified_task_fingerprint(task)
    if target.exists() and any(target.iterdir()):
        raise ExternalAgenticBackendError("external output directory must be empty")
    with tempfile.TemporaryDirectory(prefix="r3bench-agentic-") as temporary:
        temporary_root = Path(temporary)
        runtime_task = temporary_root / "task"
        scratch = temporary_root / "handoff"
        shutil.copytree(task, runtime_task)
        scratch.mkdir()
        if _verified_task_fingerprint(runtime_task) != task_fingerprint:
            raise ExternalAgenticBackendError(
                "external Agentic task snapshot differs from its source"
            )
        command = [
            str(config.executable),
            "run",
            "--protocol-version",
            config.protocol_version,
            "--task-dir",
            str(runtime_task.resolve()),
            "--output-dir",
            str(scratch),
            "--model-key",
            execution_profile.model_key,
            "--model",
            execution_profile.public_model_id,
            "--execution-profile-json",
            execution_profile.to_json(),
            "--backend",
            "harbor",
            "--environment",
            "docker",
            "--agent",
            "terminus-2",
            "--harbor-executable",
            config.harbor_executable,
            "--docker-executable",
            config.docker_executable,
            "--trajectory-format",
            config.trajectory_format,
            "--require-complete-trajectory",
        ]
        try:
            completed = subprocess.run(
                command,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                env=os.environ.copy(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExternalAgenticBackendError(
                "external Agentic process failed before a valid handoff"
            ) from exc
        if completed.returncode != 0:
            raise ExternalAgenticBackendError(
                "external Agentic process returned a non-zero status"
            )
        if (
            _verified_task_fingerprint(task) != task_fingerprint
            or _verified_task_fingerprint(runtime_task) != task_fingerprint
        ):
            raise ExternalAgenticBackendError(
                "external Agentic task changed during execution"
            )
        handoff = validate_external_backend_handoff(
            scratch,
            runtime_task,
            expected_model_key=model_key,
            expected_execution_profile=execution_profile,
        )
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(_ALLOWED_FILES):
            shutil.copy2(scratch / name, target / name)
        shutil.copytree(scratch / "artifacts", target / "artifacts")
        summary_path = target / "backend_summary.json"
        summary = _object(summary_path, "copied summary")
        summary.update(handoff)
        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        binding = target / "task_binding"
        binding.mkdir(parents=True, exist_ok=True)
        for name in sorted(_TASK_FILES):
            shutil.copy2(runtime_task / name, binding / name)
    return {
        "schema_version": "2.0",
        "status": "completed",
        "model_key": model_key,
        "public_model_id": execution_profile.public_model_id,
        "execution_profile": execution_profile.to_dict(),
        "backend": "harbor",
        "environment": "docker",
        "agent": "terminus-2",
        "external_process_started": True,
        "external_task_count": 1,
        **handoff,
        "raw_process_output_serialized": False,
        "credential_value_serialized": False,
    }


__all__ = [
    "AgenticExecutionProfile",
    "ExternalAgenticBackendConfig",
    "ExternalAgenticBackendError",
    "check_external_backend_readiness",
    "load_external_backend_config",
    "resolve_agentic_execution_profile",
    "run_external_agentic_backend",
    "validate_external_backend_handoff",
]
