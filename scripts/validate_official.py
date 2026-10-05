#!/usr/bin/env python3
"""Compile an entry with the real official libraries, without model inference."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def _source_hashes(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            files[path.relative_to(root).as_posix()] = {
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            }
    return files


async def _inspect_runtime(agent) -> dict:
    """Run official instruction/parameter conversion, never model inference."""
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.models.lite_llm import _get_completion_inputs
    from google.adk.models.llm_request import LlmRequest
    from google.adk.sessions import InMemorySessionService
    from google.adk.utils.instructions_utils import inject_session_state

    sessions = InMemorySessionService()
    # This is the harness's minimum initial state. Hints are optional, so a
    # required {hints} or another unresolved placeholder must not pass here.
    session = await sessions.create_session(
        app_name="offline_validation", user_id="offline_validation",
        state={"problem_description": "Offline instruction rendering check."},
    )
    context = ReadonlyContext(InvocationContext(
        session_service=sessions, session=session, agent=agent,
        invocation_id="offline_validation",
    ))
    instruction, bypass = await agent.canonical_instruction(context)
    if not bypass:
        instruction = await inject_session_state(instruction, context)

    request_config = agent.generate_content_config.model_copy(deep=True)
    request_config.system_instruction = instruction
    request = LlmRequest(model=agent.model.model, config=request_config)
    # A pure request conversion in the installed, version-reported ADK. This
    # private function is intentionally used to verify the real token mapping.
    _, _, _, generation = await _get_completion_inputs(request, agent.model.model)
    additional = agent.model._additional_args
    forwarded = dict(generation or {})
    for key in ("extra_body", "reasoning_effort", "seed"):
        if key in additional:
            forwarded[key] = additional[key]

    resolved = agent.generate_content_config.model_dump(mode="json", exclude_none=True)
    if forwarded.get("max_completion_tokens") != resolved.get("max_output_tokens"):
        raise ValueError("ADK did not forward max_output_tokens as max_completion_tokens")
    thinking = resolved.get("thinking_config", {})
    budget = thinking.get("thinking_budget")
    enabled = (thinking.get("include_thoughts") is not False and
               str(thinking.get("thinking_level", "")).lower() != "none" and
               budget is not None and budget > 0)
    extra = forwarded.get("extra_body", {})
    if enabled and (extra.get("thinking_token_budget") != budget or
                    extra.get("chat_template_kwargs", {}).get("enable_thinking") is not True):
        raise ValueError("Official compiler did not forward the enabled thinking budget")
    return {
        "resolved_generation": resolved,
        "forwarded_generation": forwarded,
        "instruction_rendering": {
            "status": "passed",
            "initial_state_keys": ["problem_description"],
            "rendered_characters": len(instruction),
            "rendered_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        },
    }


def validate(root: Path) -> dict:
    # macOS temporary paths start at /var, a symlink to /private/var. Match
    # validate_directory's canonical root when loading eval_config directly.
    root = root.resolve()
    # LiteLLM may otherwise fetch its model-cost map during import. This check
    # uses installed metadata only and never needs a network connection.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from adk_submission import ModelRegistry, compile_submission, validate_directory
    from adk_submission.schema import SandboxedAgentConfig
    from adk_submission.yaml_loader import load_yaml
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.models.discovery import validate_single_declared_model
    from swegemma.tools import create_tools

    limits, constraints = build_submission_limits()
    layout = validate_directory(root, limits)
    sources = _source_hashes(root)
    SandboxedAgentConfig.model_validate(load_yaml(layout.config_path, layout.root_dir, limits=limits))
    model = validate_single_declared_model(root)
    if model != "gemma-4-31b-it-qat-w4a16-ct":
        raise ValueError(f"Unsupported competition model: {model}")
    registry = ModelRegistry()
    registry.register(model, LiteLlm(model=f"openai/{model}", api_base="http://127.0.0.1:8000/v1"))
    # Official tools are closures. Bind no sandbox for this compilation-only check;
    # none are invoked, and no network/model calls or tool results are simulated.
    agent = compile_submission(
        submission_dir=root,
        tool_registry=create_tools(ctx=None),
        model_registry=registry,
        limits=limits,
        generation_constraints=constraints,
    )
    if type(agent).__name__ != "LlmAgent":
        raise ValueError("This repository's offline runtime check supports a single LlmAgent")
    runtime = asyncio.run(_inspect_runtime(agent))

    evaluation_path = root / "eval_config.yaml"
    raw_evaluation = load_yaml(evaluation_path, root, limits=limits) if evaluation_path.exists() else {}
    evaluation = raw_evaluation.get("evaluation", raw_evaluation)
    # Match the official starter's defaults and parse through the official
    # EvalConfig; placeholder data paths are never opened and no Evaluator runs.
    turns = evaluation.get("max_turns", evaluation.get("max_llm_calls"))
    config = EvalConfig(
        tasks_path=root / "__offline_tasks__.jsonl", snapshots_dir=root,
        results_dir=root, submission_dir=root, models=registry,
        timeout_seconds=int(evaluation.get("timeout_seconds", 300)),
        max_tool_calls=int(evaluation.get("max_tool_calls", 100)),
        max_time_minutes=float(evaluation.get("max_time_minutes", 60.0)),
        max_turns=int(turns) if turns is not None else None,
    )
    if _source_hashes(root) != sources:
        raise ValueError("Submission sources changed during validation; rerun the check")
    return {
        "status": "passed",
        "scope": "official CPU compilation, instruction rendering, generation forwarding and budget parsing; no inference, tool execution or scoring",
        "agent_name": agent.name,
        "agent_class": type(agent).__name__,
        "model": model,
        "tool_count": len(agent.tools),
        "source_files": sources,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "evaluation": {
            "declared": evaluation,
            "budget": asdict(config.budget),
            "harness": asdict(config.harness),
            "environment_overrides_applied": False,
        },
        **runtime,
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
