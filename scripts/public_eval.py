#!/usr/bin/env python3
"""Reproducible paired public evaluation helpers; importing needs only the stdlib.

The generated Kaggle notebook embeds this module. No test/solution patches are
used to select tasks, and no benchmark answers are passed to the submitted agent.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path


def select_tasks(tasks, count=8, seed=20261004, task_ids=(), exclude_ids=()):
    """Sample by repository, approximately proportional to public pool size.

    Give each repository one slot when possible, then allocate the rest by its
    representation deficit. Hash ordering avoids first-ID bias and depends only
    on the recorded seed, repository and ID, never on solutions or outcomes.
    """
    by_id = {task.instance_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task IDs in public pool")
    if task_ids:
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Duplicate requested task IDs")
        unknown = set(task_ids) - by_id.keys()
        if unknown:
            raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
        if set(task_ids) & set(exclude_ids):
            raise ValueError("Requested task IDs overlap excluded tuning tasks")
        return [by_id[task_id] for task_id in task_ids]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Task count must be a positive integer")
    pools = defaultdict(list)
    excluded = set(exclude_ids)
    for task in tasks:
        if task.instance_id not in excluded:
            pools[task.repo].append(task)
    total = sum(map(len, pools.values()))
    if total == 0:
        raise ValueError("No eligible public tasks")
    count = min(count, total)

    def order(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    repos = sorted(pools, key=order)
    for repo in repos:
        pools[repo].sort(key=lambda task: order(task.instance_id))
    quotas = {repo: 0 for repo in repos}
    if count >= len(repos):
        quotas = {repo: 1 for repo in repos}
    for _ in range(count - sum(quotas.values())):
        repo = max((repo for repo in repos if quotas[repo] < len(pools[repo])),
                   key=lambda repo: (count * len(pools[repo]) / total - quotas[repo], order(repo)))
        quotas[repo] += 1
    # Interleave repositories so partial budget-limited runs keep some coverage.
    chosen = []
    for index in range(max(quotas.values())):
        chosen.extend(pools[repo][index] for repo in repos if index < quotas[repo])
    return chosen


def wilson_interval(resolved, count):
    """Two-sided 95% binomial Wilson interval; descriptive, not a hidden score."""
    if not count:
        return None
    z = 1.959963984540054
    rate = resolved / count
    divisor = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / divisor
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / divisor
    return [max(0.0, center - half), min(1.0, center + half)]


def failure_category(row):
    """Best-effort diagnostic tags; raw logs remain authoritative."""
    if row.get("resolved"):
        return "resolved"
    error = (row.get("error_message") or "").lower()
    output = (row.get("test_output") or "").lower()
    if "snapshot file not found" in error or "missing test specification" in error:
        return "missing_evaluation_data"
    if "failed to apply agent patch" in error:
        return "patch_did_not_apply"
    if "failed to apply test_patch" in error:
        return "verification_setup"
    if any(token in error for token in ("budget", "max_turns", "max_tool_calls")):
        return "agent_budget"
    if "timeout" in error or "timed out" in error:
        return "timeout"
    if row.get("exception_type") or "evaluation error" in error:
        return "runtime_error"
    if "modulenotfounderror" in output or "importerror" in output:
        return "test_import_error"
    if not (row.get("agent_patch") or "").strip():
        return "empty_patch"
    return "tests_failed" if row.get("test_exit_code") else "verification_failed"


def summarize(rows, task_ids, variant_names, seed=20261004):
    """Summarize observed runs; compare only complete baseline/candidate pairs."""
    output = {"planned_task_count": len(task_ids), "variants": {}, "paired": None,
              "interpretation": "Public development diagnostics, not a Kaggle score. "
              "Intervals are descriptive binomial intervals; repository/task dependence, "
              "stratification and tuning selection can make them overconfident."}
    for variant in variant_names:
        completed = [row for row in rows if row["variant"] == variant]
        successes = sum(bool(row["resolved"]) for row in completed)
        repos = defaultdict(list)
        for row in completed:
            repos[row["repo"]].append(row)
        output["variants"][variant] = {
            "completed": len(completed), "resolved": successes,
            "resolution_rate": successes / len(completed) if completed else None,
            "wilson_95": wilson_interval(successes, len(completed)),
            "duration_seconds": sum(row.get("duration_seconds", 0) for row in completed),
            "failure_categories": dict(Counter(row["failure_category"] for row in completed)),
            "per_repository": {repo: {"completed": len(items),
                                     "resolved": sum(bool(item["resolved"]) for item in items)}
                               for repo, items in sorted(repos.items())},
        }
    if "baseline" in variant_names and "candidate" in variant_names:
        lookup = {(row["variant"], row["id"]): row for row in rows}
        paired_ids = [task_id for task_id in task_ids
                      if ("baseline", task_id) in lookup and ("candidate", task_id) in lookup]
        deltas = [int(lookup["candidate", task_id]["resolved"])
                  - int(lookup["baseline", task_id]["resolved"]) for task_id in paired_ids]
        wins = [key for key, delta in zip(paired_ids, deltas) if delta == 1]
        regressions = [key for key, delta in zip(paired_ids, deltas) if delta == -1]
        # A nonparametric bootstrap degenerates with zero discordant pairs.
        # Exact sign-test p-value remains honest in that case (p=1).
        discordant = len(wins) + len(regressions)
        smaller = min(len(wins), len(regressions))
        p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(smaller + 1))
                      / 2 ** discordant) if discordant else 1.0
        interval = None
        if deltas and discordant:
            rng = random.Random(seed)
            bootstrap = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
                               for _ in range(5000))
            interval = [bootstrap[124], bootstrap[4874]]
        output["paired"] = {
            "completed_pairs": len(paired_ids), "wins": wins, "regressions": regressions,
            "tied": len(paired_ids) - discordant,
            "candidate_minus_baseline": sum(deltas) / len(deltas) if deltas else None,
            "paired_bootstrap_95": interval, "exact_discordance_p_two_sided": p_value,
            "note": "Small samples and few wins/regressions do not establish a 0.03 gain. "
                    "Bootstrap interval is omitted if no pairs differ; use fresh holdout tasks.",
        }
    return output


def run_sync(function, **kwargs):
    """Run the harness coroutine in either a notebook or a plain Python process."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(function(**kwargs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(function(**kwargs))).result()


def write_json(path, data):
    """Atomic replacement keeps completed records readable after interruptions."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def public_evaluator_class(base):
    """Keep the official pipeline, with equal grading time and public inputs only.

    One evaluator is used sequentially. Config is restored even on exceptions;
    this must not be used to run concurrent tasks on the same instance.
    """
    class PublicEvaluator(base):
        def __init__(self, config, verification_timeout_seconds):
            super().__init__(config)
            self.verification_timeout_seconds = verification_timeout_seconds

        def _get_secret_bundle_data(self):
            return {}, {}, None

        async def _run_agent_sandbox(self, *args, **kwargs):
            result = await super()._run_agent_sandbox(*args, **kwargs)
            self.config = replace(self.config, timeout_seconds=self.verification_timeout_seconds)
            self.docker.timeout_seconds = self.verification_timeout_seconds
            return result

        async def evaluate_task(self, *args, **kwargs):
            original_config = self.config
            original_timeout = self.docker.timeout_seconds
            try:
                return await super().evaluate_task(*args, **kwargs)
            finally:
                self.config = original_config
                self.docker.timeout_seconds = original_timeout

    return PublicEvaluator


def evaluate_variants(*, selected_tasks, variants, models, adapters, data_dir,
                      output_dir, task_file_sha256, seed=20261004,
                      max_run_minutes=180, task_time_cap_minutes=None,
                      verification_timeout_seconds=300, helper_sha256=None):
    """Evaluate fresh task sandboxes with each archive's own settings.

    The run budget stops scheduling new pairs, not a running task or verification.
    A cap changes the evaluation protocol and is recorded explicitly. Results and
    traces are persisted after every task; resume requires identical run identity.
    """
    import importlib.metadata
    import yaml
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    if not selected_tasks:
        raise ValueError("No selected tasks")
    if not variants:
        raise ValueError("No archive variants")
    if type(verification_timeout_seconds) is not int or verification_timeout_seconds <= 0:
        raise ValueError("verification_timeout_seconds must be a positive integer")
    for label, value in (("max_run_minutes", max_run_minutes),
                         ("task_time_cap_minutes", task_time_cap_minutes)):
        if value is not None and (not isinstance(value, (int, float)) or value <= 0
                                  or not math.isfinite(value)):
            raise ValueError(f"{label} must be positive and finite or None")
    budgets = {}
    for name, variant in variants.items():
        config_file = Path(variant["directory"]) / "eval_config.yaml"
        raw = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
        settings = (raw or {}).get("evaluation", {})
        budgets[name] = {key: settings[key] for key in
                         ("timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns")
                         if key in settings}
        if task_time_cap_minutes is not None:
            budgets[name]["max_time_minutes"] = min(
                budgets[name].get("max_time_minutes", 60.0), task_time_cap_minutes)
    manifest = {
        "protocol_version": 2, "seed": seed, "helper_sha256": helper_sha256,
        "tasks": [{"id": task.instance_id, "repo": task.repo} for task in selected_tasks],
        "task_file_sha256": task_file_sha256,
        "archives": {name: variant["sha256"] for name, variant in variants.items()},
        "effective_budgets": budgets, "task_time_cap_minutes": task_time_cap_minutes,
        "verification_timeout_seconds": verification_timeout_seconds,
        "secret_bundle_hydration": False,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("swegemma", "adk-submission", "adk-eval-core", "google-adk", "vllm")},
        "compaction": {"compaction_interval": 5, "overlap_size": 2,
                       "token_threshold": 14336, "event_retention_size": 5},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = output_dir / "run-manifest.json"
    if identity.exists():
        if json.loads(identity.read_text()) != manifest:
            raise ValueError("Existing run differs in archive, tasks, packages or settings. "
                             "Choose a new RUN_LABEL; do not mix evaluations.")
    else:
        write_json(identity, manifest)
    records_dir = output_dir / "records"
    records_dir.mkdir(exist_ok=True)
    records = []
    expected = {(name, task.instance_id) for name in variants for task in selected_tasks}
    seen = set()
    for path in sorted(records_dir.glob("*.json")):
        row = json.loads(path.read_text())
        key = (row["variant"], row["id"])
        if key not in expected or key in seen or row["archive_sha256"] != variants[key[0]]["sha256"]:
            raise ValueError(f"Unexpected or duplicate cached task record: {path}")
        if row.get("exception_type"):
            raise ValueError(f"Previous run had a runtime exception in {path}; inspect its traceback "
                             "and use a new RUN_LABEL after fixing the cause.")
        seen.add(key)
        records.append(row)
    done = {(row["variant"], row["id"]) for row in records}
    task_ids = [task.instance_id for task in selected_tasks]
    limits, gen_constraints = build_submission_limits()
    evaluators = {}
    PublicEvaluator = public_evaluator_class(Evaluator)
    for name, variant in variants.items():
        config = EvalConfig(
            tasks_path=data_dir / "tasks.jsonl", snapshots_dir=data_dir / "snapshots",
            results_dir=output_dir / name, submission_dir=Path(variant["directory"]),
            models=models, sandbox="subprocess", **budgets[name],
            limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(**manifest["compaction"]),
            graph_dir=str(data_dir / "graphs"), embeddings_dir=str(data_dir / "embeddings"),
            wheels_dir=data_dir / "wheels", verbose=False,
        )
        evaluators[name] = PublicEvaluator(config, verification_timeout_seconds)
    start = time.monotonic()
    for index, task in enumerate(selected_tasks, start=1):
        if max_run_minutes is not None and (time.monotonic() - start) / 60 >= max_run_minutes:
            print("Run time allowance reached; saved completed tasks. Resume this RUN_LABEL to continue.")
            break
        # Alternate which archive runs first to reduce fixed ordering effects.
        names = list(variants) if index % 2 else list(reversed(variants))
        for name in names:
            if (name, task.instance_id) in done:
                print(f"[{index}/{len(selected_tasks)}] {name} {task.instance_id}: cached")
                continue
            print(f"[{index}/{len(selected_tasks)}] {name} {task.instance_id}", flush=True)
            task_start = time.monotonic()
            failure = None
            try:
                result = run_sync(evaluators[name].evaluate_task, task=task,
                                  task_index=index, total_tasks=len(selected_tasks))
                row = result.model_dump(mode="json", exclude={"trace"})
            except Exception as error:
                failure = error
                row = {"resolved": False, "test_exit_code": -1, "agent_patch": "",
                       "error_message": str(error), "exception_type": type(error).__name__,
                       "traceback": traceback.format_exc(),
                       "duration_seconds": time.monotonic() - task_start}
            row.update(id=task.instance_id, repo=task.repo, variant=name,
                       archive_sha256=variants[name]["sha256"])
            row["failure_category"] = failure_category(row)
            key = hashlib.sha256(f"{name}:{task.instance_id}".encode()).hexdigest()[:24]
            write_json(records_dir / f"{key}.json", row)
            records.append(row)
            done.add((name, task.instance_id))
            summary = summarize(records, task_ids, list(variants), seed)
            write_json(output_dir / "development-summary.json", summary)
            print(f"  {row['failure_category']}, {row.get('tool_calls', 0)} tool calls, "
                  f"{row.get('duration_seconds', 0):.1f}s", flush=True)
            if failure is not None:
                raise RuntimeError(f"Evaluation stopped after a runtime exception; full traceback: "
                                   f"{records_dir / f'{key}.json'}") from failure
    summary = summarize(records, task_ids, list(variants), seed)
    write_json(output_dir / "development-summary.json", summary)
    return summary, records
