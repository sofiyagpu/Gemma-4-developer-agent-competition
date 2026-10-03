#!/usr/bin/env python3
"""Compile an entry with the real official libraries, without model inference."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def validate(root: Path) -> dict:
    from adk_submission import ModelRegistry, compile_submission, validate_directory
    from adk_submission.schema import SandboxedAgentConfig
    from adk_submission.yaml_loader import load_yaml
    from swegemma.config import build_submission_limits
    from swegemma.models.discovery import validate_single_declared_model
    from swegemma.tools import create_tools

    limits, constraints = build_submission_limits()
    layout = validate_directory(root, limits)
    SandboxedAgentConfig.model_validate(load_yaml(layout.config_path, layout.root_dir, limits=limits))
    model = validate_single_declared_model(root)
    if model != "gemma-4-31b-it-qat-w4a16-ct":
        raise ValueError(f"Unsupported competition model: {model}")
    registry = ModelRegistry()
    registry.register(model, model)
    # Official tools are closures. Bind no sandbox for this compilation-only check;
    # none are invoked, and no network/model calls or tool results are simulated.
    agent = compile_submission(
        submission_dir=root,
        tool_registry=create_tools(ctx=None),
        model_registry=registry,
        limits=limits,
        generation_constraints=constraints,
    )
    return {
        "status": "passed",
        "scope": "official CPU directory/schema/tool/agent compilation; no inference or scoring",
        "agent_name": agent.name,
        "agent_class": type(agent).__name__,
        "model": model,
        "tool_count": len(agent.tools),
        "versions": {name: importlib.metadata.version(name) for name in (
            "adk-submission", "adk-eval-core", "swegemma", "google-adk", "google-genai"
        )},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "submission")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.submission_dir.resolve())
    report = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
