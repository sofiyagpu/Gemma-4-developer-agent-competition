#!/usr/bin/env python3
"""Offline checks for this repository's single-agent submission subset.

This is not the official adk-submission compiler and does not measure quality.
Only declarative, referenced agent/prompt/generation/budget files are supported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML is required: python -m pip install -r requirements-dev.txt")


MODEL = "gemma-4-31b-it-qat-w4a16-ct"
TOOLS = frozenset({
    "run_command", "submit_patch", "get_status", "read_file", "edit_file",
    "write_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph",
})
MAX_TOTAL_BYTES = 3 * 1024**3
MAX_YAML_BYTES = 50 * 1024**2
MAX_INSTRUCTION_CHARS = 1_000_000
REPO_ROOT = Path(__file__).resolve().parents[1]


class ValidationError(ValueError):
    """The submission does not meet this project's supported contract."""


@dataclass
class ValidatedSubmission:
    # Package this checked byte snapshot, without rereading mutable source files.
    files: dict[str, bytes]
    agent: dict[str, Any]
    evaluation: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "status": "passed",
            "check_scope": "local supported-subset checks; not official harness validation",
            "model": self.agent["model"],
            "tool_count": len(self.agent["tools"]),
            "files": sorted(self.files),
            "unpacked_bytes": sum(map(len, self.files.values())),
            "evaluation": self.evaluation,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _safe_relative_path(name: str) -> str:
    _require(isinstance(name, str) and bool(name), "Include path must be a nonempty string")
    _require("\x00" not in name and "\\" not in name, f"Unsafe path: {name!r}")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and not PureWindowsPath(name).drive,
             f"Absolute paths are forbidden: {name!r}")
    _require(all(part not in ("", ".", "..") for part in name.split("/")),
             f"Traversal or noncanonical path is forbidden: {name!r}")
    _require(all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in path.parts),
             f"Unsupported filename: {name!r}")
    return path.as_posix()


def _allowed_file(name: str) -> bool:
    path = PurePosixPath(name)
    return name in {"agent.yaml", "eval_config.yaml"} or (
        len(path.parts) >= 2 and (
            (path.parts[0] == "prompts" and path.suffix in {".md", ".txt"}) or
            (path.parts[0] == "configs" and path.suffix in {".yaml", ".yml"})
        )
    )


def _read_files(root: Path) -> dict[str, bytes]:
    _require(not root.is_symlink(), "Submission root must not be a symlink")
    _require(root.is_dir(), f"Submission directory does not exist: {root}")
    files: dict[str, bytes] = {}
    total = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            path = Path(current) / name
            relative = _safe_relative_path(path.relative_to(root).as_posix())
            mode = path.lstat().st_mode
            _require(not stat.S_ISLNK(mode), f"Symlinks are forbidden: {relative}")
            if stat.S_ISDIR(mode):
                _require(PurePosixPath(relative).parts[0] in {"prompts", "configs"},
                         f"Unexpected directory: {relative}")
                continue
            _require(stat.S_ISREG(mode), f"Only regular files are allowed: {relative}")
            _require(_allowed_file(relative), f"File is not in the submission allowlist: {relative}")
            size = path.stat().st_size
            # This source-only subset has no adapters; no individual file needs 50 MiB.
            _require(size <= MAX_YAML_BYTES, f"File exceeds local 50 MiB limit: {relative}")
            total += size
            _require(total < MAX_TOTAL_BYTES, "Unpacked submission must be smaller than 3 GiB")
            data = path.read_bytes()
            _require(len(data) == size, f"File changed while validating: {relative}; retry")
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValidationError(f"File must be UTF-8: {relative}") from exc
            _require(b"\x00" not in data, f"NUL bytes are forbidden: {relative}")
            files[relative] = data
            _require(len(files) <= 10_000, "Submission exceeds 10,000 files")
    _require("agent.yaml" in files, "Required root agent.yaml is missing")
    _require(sum(n.endswith((".yaml", ".yml")) for n in files) <= 1_000,
             "Submission exceeds 1,000 YAML files")
    return dict(sorted(files.items()))


class _Resolver:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.used: set[str] = set()
        self.expanded_bytes = 0

    def load(self, name: str, stack: tuple[str, ...] = ()) -> Any:
        name = _safe_relative_path(name)
        _require(name in self.files, f"Included file does not exist: {name}")
        _require(name not in stack, f"Include cycle: {' -> '.join((*stack, name))}")
        _require(len(stack) <= 10, "Include depth exceeds 10")
        self.used.add(name)
        self.expanded_bytes += len(self.files[name])
        _require(self.expanded_bytes <= MAX_YAML_BYTES, "Cumulative includes exceed 50 MiB")
        text = self.files[name].decode("utf-8")
        if name.endswith((".md", ".txt")):
            return text

        resolver = self

        class Loader(yaml.SafeLoader):
            depth = 0

            def compose_node(self, parent: Any, index: Any) -> Any:
                event = self.peek_event()
                _require(not isinstance(event, yaml.AliasEvent) and not getattr(event, "anchor", None),
                         f"YAML anchors and aliases are unsupported: {name}")
                self.depth += 1
                _require(self.depth <= 50, f"YAML nesting exceeds 50: {name}")
                try:
                    return super().compose_node(parent, index)
                finally:
                    self.depth -= 1

            def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
                _require(isinstance(node, yaml.MappingNode), f"Expected YAML mapping: {name}")
                mapping: dict[str, Any] = {}
                for key_node, value_node in node.value:
                    key = self.construct_object(key_node, deep=deep)
                    _require(isinstance(key, str), f"YAML mapping keys must be strings: {name}")
                    _require(key not in mapping, f"Duplicate YAML key {key!r}: {name}")
                    mapping[key] = self.construct_object(value_node, deep=deep)
                return mapping

        def include(loader: Loader, node: Any) -> Any:
            _require(isinstance(node, yaml.ScalarNode), f"!include needs one path: {name}")
            target = loader.construct_scalar(node)
            # The competition compiler resolves includes relative to the submission root.
            return resolver.load(target, (*stack, name))

        Loader.add_constructor("!include", include)
        try:
            return yaml.load(text, Loader=Loader)
        except yaml.YAMLError as exc:
            raise ValidationError(f"Invalid YAML in {name}: {exc}") from exc


def _mapping(value: Any, path: str, allowed: set[str]) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{path} must be a mapping")
    _require(not (set(value) - allowed), f"Unsupported fields in {path}: {sorted(set(value) - allowed)}")
    return value


def _number(value: Any, path: str, minimum: float | None = None,
            maximum: float | None = None, integer: bool = False) -> None:
    _require(type(value) in ({int} if integer else {int, float}), f"{path} must be {'an integer' if integer else 'a number'}")
    # math.isfinite(int) can overflow for maliciously enormous integers.
    _require(not isinstance(value, float) or math.isfinite(value), f"{path} must be finite")
    _require(minimum is None or value >= minimum, f"{path} must be >= {minimum}")
    _require(maximum is None or value <= maximum, f"{path} must be <= {maximum}")


def _generation(value: Any) -> None:
    path = "generate_content_config"
    config = _mapping(value, path, {
        "temperature", "top_p", "top_k", "max_output_tokens", "presence_penalty",
        "frequency_penalty", "stop_sequences", "response_mime_type", "seed", "thinking_config",
    })
    bounds = {
        "temperature": (0, None, False), "top_p": (0, 1, False),
        "top_k": (1, None, True), "max_output_tokens": (1, 32768, True),
        "presence_penalty": (None, None, False), "frequency_penalty": (None, None, False),
        "seed": (None, None, True),
    }
    for key, (minimum, maximum, integer) in bounds.items():
        if key in config:
            _number(config[key], f"{path}.{key}", minimum, maximum, integer)
    if "stop_sequences" in config:
        _require(isinstance(config["stop_sequences"], list) and
                 all(isinstance(item, str) for item in config["stop_sequences"]),
                 f"{path}.stop_sequences must be a list of strings")
    if "response_mime_type" in config:
        _require(isinstance(config["response_mime_type"], str) and bool(config["response_mime_type"]),
                 f"{path}.response_mime_type must be a nonempty string")
    if "thinking_config" in config:
        thinking = _mapping(config["thinking_config"], f"{path}.thinking_config",
                            {"thinking_budget", "include_thoughts", "thinking_level"})
        if "thinking_budget" in thinking:
            _number(thinking["thinking_budget"], "thinking_budget", 0, 32768, True)
        if "include_thoughts" in thinking:
            _require(type(thinking["include_thoughts"]) is bool, "include_thoughts must be a boolean")
        if "thinking_level" in thinking:
            level = thinking["thinking_level"]
            _require(isinstance(level, str) and level.upper() in {"MINIMAL", "LOW", "MEDIUM", "HIGH", "NONE"},
                     "thinking_level must be MINIMAL, LOW, MEDIUM, HIGH, or NONE")


def validate_submission(root: Path | str) -> ValidatedSubmission:
    files = _read_files(Path(root))
    resolver = _Resolver(files)
    agent = _mapping(resolver.load("agent.yaml"), "agent.yaml", {
        "agent_class", "name", "model", "description", "instruction", "global_instruction",
        "tools", "include_contents", "generate_content_config",
    })
    _require(agent.get("agent_class", "LlmAgent") == "LlmAgent", "Only a single LlmAgent is supported")
    _require(isinstance(agent.get("name"), str) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", agent["name"])),
             "Agent name must be a Python-style identifier")
    _require(agent.get("model") == MODEL, f"Competition model must be {MODEL}")
    _require(isinstance(agent.get("instruction"), str) and bool(agent["instruction"].strip()),
             "Agent instruction must be a nonempty string")
    for key in ("instruction", "global_instruction", "description"):
        if key in agent:
            _require(isinstance(agent[key], str) and len(agent[key]) <= MAX_INSTRUCTION_CHARS,
                     f"{key} must be a string of at most 1,000,000 characters")
    tool_names = agent.get("tools")
    _require(isinstance(tool_names, list) and all(isinstance(name, str) for name in tool_names),
             "tools must be a list of built-in tool names")
    _require(len(tool_names) == len(TOOLS) and set(tool_names) == TOOLS,
             "This submission must declare exactly the nine built-in tools, once each")
    if "include_contents" in agent:
        _require(agent["include_contents"] in ("default", "none"), "include_contents must be default or none")
    if "generate_content_config" in agent:
        _generation(agent["generate_content_config"])

    evaluation: dict[str, Any] = {}
    if "eval_config.yaml" in files:
        config = _mapping(resolver.load("eval_config.yaml"), "eval_config.yaml", {"evaluation"})
        _require("evaluation" in config, "eval_config.yaml needs an evaluation mapping")
        evaluation = _mapping(config["evaluation"], "evaluation", {
            "timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns",
        })
        for key, value in evaluation.items():
            _number(value, f"evaluation.{key}", integer=key != "max_time_minutes")
            _require(value > 0, f"evaluation.{key} must be greater than zero")

    unused = set(files) - resolver.used
    _require(not unused, f"Unreferenced submission files are forbidden: {sorted(unused)}")
    return ValidatedSubmission(files, agent, evaluation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", nargs="?", type=Path, default=REPO_ROOT / "submission")
    args = parser.parse_args()
    try:
        result = validate_submission(args.submission_dir)
    except (ValidationError, OSError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
