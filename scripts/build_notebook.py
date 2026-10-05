#!/usr/bin/env python3
"""Build a self-contained Kaggle notebook for paired public evaluation."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

try:
    from .package_submission import package_submission
except ImportError:
    from package_submission import package_submission

ROOT = Path(__file__).resolve().parents[1]


def build_notebook(archive: Path, baseline: Path | None, output: Path) -> None:
    archives = {"candidate": archive}
    if baseline is not None:
        archives = {"baseline": baseline, **archives}
    entries = {}
    for name, path in archives.items():
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        manifest = path.with_suffix('.manifest.json')
        if manifest.exists() and json.loads(manifest.read_text())["archive"]["sha256"] != digest:
            raise ValueError(f"Archive does not match its manifest: {path}. Rebuild it before embedding.")
        entries[name] = {"sha256": digest, "base64": base64.b64encode(payload).decode()}
    starter = json.loads((ROOT / "reference/official-getting-started.ipynb").read_text())
    helper_source = (ROOT / "scripts/public_eval.py").read_text()
    cells = []

    def add(kind: str, source: str) -> None:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# Gemma 4: paired baseline/candidate evaluation

This notebook restores the exact agent archives, compiles them with the official harness,
and optionally compares them on the **same public tasks**. It does not submit to Kaggle.
The baseline is the preserved entry reported as 0.10 by its author. The candidate's
leaderboard score is unknown; a public development result does not establish a hidden score.

Attach the **Gemma 4 Developer Agent competition**, the latest
**metric/gemma-4-developer-agent-wheelhouse** dataset, and
**google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2** model.
For evaluation choose **GPU L4 x4**, **Internet off**, and start a fresh session after
changing the wheelhouse. By default only CPU compilation runs.

Runtime installation and model serving are adapted from [Ryan Holbrook's official starter,
version 2](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent).
""")
    add("code", f"""RUN_EVALUATION = False  # Enable only to run real tasks on the GPU.
RUN_VARIANTS = {list(archives)!r}
RUN_LABEL = 'paired-v3'  # New label for changed archives, tasks or evaluation settings.
TASK_IDS = []  # Explicit IDs reproduce a previous panel exactly.
TASK_COUNT = 8  # Diagnostic smoke panel; increase for broader evidence.
SELECTION_SEED = 20261004
EXCLUDE_TASK_IDS = []  # Previously tuned-on IDs; exclude for a fresh holdout panel.
MAX_RUN_MINUTES = 180  # Stop before the next pair; in-flight tasks can exceed this.
TASK_TIME_CAP_MINUTES = None  # None preserves each archive's real task budget.
VERIFICATION_TIMEOUT_SECONDS = 300  # Identical public grading allowance for both archives.
# A numeric cap is a cheaper smoke protocol, not an evaluation of the actual archive budgets.
""")
    add("markdown", "## Install the official offline runtime")
    install = "".join(starter["cells"][2]["source"])
    install = install.replace(
        "WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')",
        """wheelhouse_candidates = [
    Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
    Path('/kaggle/input/gemma-4-developer-agent-wheelhouse'),
]
WHEELHOUSE_DIR = next((p for p in wheelhouse_candidates if any(p.glob('*.whl'))), None)
if WHEELHOUSE_DIR is None:
    raise FileNotFoundError('Attach metric/gemma-4-developer-agent-wheelhouse as notebook input. '
                            'No wheel files were found in either supported Kaggle mount.')""")
    # A fresh directory avoids stale wheel versions after a dataset update.
    install = install.replace("import sys\n", "import sys\nimport tempfile\n")
    install = install.replace("tmp_whl = Path('/tmp/wheelhouse')", "tmp_whl = Path(tempfile.mkdtemp(prefix='gemma-wheelhouse-'))")
    add("code", install)
    cells[-1]["metadata"]["tags"] = ["kaggle-runtime-install"]
    add("markdown", "## Restore and compile both exact archives")
    add("code", f'''import base64
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

ARCHIVES = {entries!r}
WORKING_DIR = Path(os.environ.get('GEMMA_WORKING_DIR', '/kaggle/working'))
WORKING_DIR.mkdir(parents=True, exist_ok=True)
VARIANTS = {{}}
assert RUN_VARIANTS and len(set(RUN_VARIANTS)) == len(RUN_VARIANTS)
for name in RUN_VARIANTS:
    entry = ARCHIVES[name]
    payload = base64.b64decode(entry['base64'])
    assert hashlib.sha256(payload).hexdigest() == entry['sha256']
    archive_name = 'submission.zip' if name == 'candidate' else 'submission-baseline-0.10.zip'
    (WORKING_DIR / archive_name).write_bytes(payload)
    directory = Path(tempfile.mkdtemp(prefix=f'{{name}}-', dir=WORKING_DIR))
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert 'agent.yaml' in zf.namelist()
        for item in zf.infolist():
            assert not Path(item.filename).is_absolute() and '..' not in Path(item.filename).parts
            assert (item.external_attr >> 16) & 0o170000 != 0o120000, 'Archive contains symlink'
        zf.extractall(directory)
    VARIANTS[name] = {{'directory': directory, 'sha256': entry['sha256']}}
    print(name, entry['sha256'])
AGENT_DIR = VARIANTS.get('candidate', next(iter(VARIANTS.values())))['directory']
''')
    add("code", """from adk_submission import ModelRegistry, compile_submission, validate_directory
from adk_submission.schema import SandboxedAgentConfig
from adk_submission.yaml_loader import load_yaml
from swegemma.config import build_submission_limits
from swegemma.models.discovery import validate_single_declared_model
from swegemma.tools import create_tools

limits, gen_constraints = build_submission_limits()
declared_models = set()
for name, variant in VARIANTS.items():
    directory = variant['directory']
    layout = validate_directory(directory, limits)
    parsed = load_yaml(layout.config_path, layout.root_dir, limits=limits)
    SandboxedAgentConfig.model_validate(parsed)
    declared_model = validate_single_declared_model(directory)
    declared_models.add(declared_model)
    compile_models = ModelRegistry()
    compile_models.register(declared_model, declared_model)
    compiled = compile_submission(
        submission_dir=directory, tool_registry=create_tools(ctx=None),
        model_registry=compile_models, limits=limits,
        generation_constraints=gen_constraints,
    )
    print('Official CPU compilation passed:', name, compiled.name, declared_model)
assert len(declared_models) == 1, 'Paired archives must use the same competition base model'
# This comparison reuses one model server. The preserved baseline has no adapters.
if 'baseline' in VARIANTS and 'candidate' in VARIANTS:
    baseline_adapters = VARIANTS['baseline']['directory'] / 'adapters'
    assert not baseline_adapters.exists(), 'Baseline adapters require separate model serving'
print('GPU evaluation enabled:', RUN_EVALUATION)
""")
    add("markdown", """## Public evaluation helpers

Selection is seeded and stratified by repository, approximately in proportion to its
public task count. It never reads reference solutions or verification outcomes. The
panel and archive hashes are saved. A small panel can diagnose broken tools and empty
patches; it cannot reliably distinguish leaderboard scores of 0.10 and 0.13.

Each task gets a fresh harness sandbox. We alternate archive order and preserve archive
budgets. Every outcome, patch, error, test output and official trace path is saved before
starting the next task. Rerunning with the same label resumes completed tasks; changed
inputs/settings require a new label. Retrying failed tasks also requires a new label.
""")
    add("code", helper_source)
    add("code", f"""HELPER_SHA256 = {hashlib.sha256(helper_source.encode()).hexdigest()!r}
if RUN_EVALUATION:
    from swegemma.models import load_tasks
    from swegemma.deduplication import resolve_task_snapshot_paths
    DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
    if not (DATA_DIR / 'tasks.jsonl').is_file():
        DATA_DIR = Path('/kaggle/input/gemma-4-developer-agent')
    TASKS_PATH = DATA_DIR / 'tasks.jsonl'
    if not TASKS_PATH.is_file():
        raise FileNotFoundError('Attach the Gemma 4 Developer Agent competition data; tasks.jsonl is missing.')
    tasks = load_tasks(TASKS_PATH)
    selected_tasks = select_tasks(tasks, TASK_COUNT, SELECTION_SEED, TASK_IDS, EXCLUDE_TASK_IDS)
    assert RUN_LABEL and Path(RUN_LABEL).name == RUN_LABEL and RUN_LABEL not in ('.', '..')
    RESULTS_DIR = WORKING_DIR / 'public-evaluation' / RUN_LABEL
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selection = {{'seed': SELECTION_SEED, 'pool_count': len(tasks),
                 'excluded_ids': list(EXCLUDE_TASK_IDS),
                 'task_file_sha256': hashlib.sha256(TASKS_PATH.read_bytes()).hexdigest(),
                 'helper_sha256': HELPER_SHA256,
                 'selected': [{{'id': task.instance_id, 'repo': task.repo}} for task in selected_tasks]}}
    print(json.dumps(selection, indent=2))
    # Fail before model startup if public verification would be impossible.
    missing_tests = [task.instance_id for task in selected_tasks
                     if not (task.test_patch or task.FAIL_TO_PASS or task.PASS_TO_PASS)]
    missing_snapshots = [task.instance_id for task in selected_tasks
                        if not resolve_task_snapshot_paths(DATA_DIR / 'snapshots', task.instance_id, task.repo)[0].exists()]
    if missing_tests or missing_snapshots:
        raise RuntimeError(f'Public evaluation inputs missing: test specifications={{missing_tests}}, '
                           f'snapshots={{missing_snapshots}}. No GPU evaluation started; '
                           'attach the public development data with test specifications.')
    selection_path = RESULTS_DIR / 'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Selection/helper changed: choose a new RUN_LABEL.')
    write_json(selection_path, selection)
""")
    server = "".join(starter["cells"][8]["source"])
    server = server.replace("gpu_memory_utilization=0.90", "gpu_memory_utilization=0.80")
    server = server.replace("INFERENCE_API_KEY = 'EMPTY'", """if not (MODEL_PATH / 'config.json').is_file():
    alternatives = list(Path('/kaggle/input').glob('**/gemma-4-31b-it-qat-w4a16-ct/*/config.json'))
    if len(alternatives) != 1:
        raise FileNotFoundError('Attach google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2; model config is missing or ambiguous.')
    MODEL_PATH = alternatives[0].parent
INFERENCE_API_KEY = 'EMPTY'""")
    server = server.replace("gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1",
                            "gpu_count = torch.cuda.device_count()\nassert gpu_count == 4, 'Select GPU L4 x4 for the competition model'")
    evaluate = """summary, records = evaluate_variants(
    selected_tasks=selected_tasks, variants=VARIANTS, models=models, adapters=adapters,
    data_dir=DATA_DIR, output_dir=RESULTS_DIR,
    task_file_sha256=selection['task_file_sha256'], seed=SELECTION_SEED,
    max_run_minutes=MAX_RUN_MINUTES, task_time_cap_minutes=TASK_TIME_CAP_MINUTES,
    verification_timeout_seconds=VERIFICATION_TIMEOUT_SECONDS, helper_sha256=HELPER_SHA256,
)
import pandas as pd
columns = ['variant', 'id', 'resolved', 'failure_category', 'test_exit_code',
           'duration_seconds', 'tool_calls', 'total_llm_calls', 'error_message']
display(pd.DataFrame(records).reindex(columns=columns))
print(json.dumps(summary, indent=2))
print('Full artifacts:', RESULTS_DIR)
"""
    add("code", "server_instance = None\nif RUN_EVALUATION:\n    try:\n"
        + "".join("        " + line + "\n" for line in (server + "\n\n" + evaluate).splitlines())
        + "    except Exception:\n        import traceback\n        error_path = WORKING_DIR / 'runtime-error.txt'\n        error_path.write_text(traceback.format_exc())\n        print('Full exception saved to:', error_path)\n        raise\n"
        + "    finally:\n        if server_instance is not None:\n            server_instance.stop()\n")
    add("markdown", """## Read the comparison and download the candidate

`public-evaluation/<RUN_LABEL>/development-summary.json` reports both resolution rates,
descriptive Wilson intervals, paired wins/regressions, a paired bootstrap interval and
an exact discordance test. Infrastructure errors remain in the denominator and are
identified separately in the saved records. Inspect regressions and traces before
changing the prompt; then test on fresh IDs by filling `EXCLUDE_TASK_IDS`.

The time allowance only prevents starting another pair. Agent task time excludes some
setup/verification overhead; `timeout_seconds` is a **command** timeout. Plan GPU time
accordingly. More public tasks and fresh holdout results are stronger evidence than a
small tuned panel, but only a competition submission establishes a leaderboard score.
Public verification uses the same 300-second test timeout for both variants, independent
of their agent command timeouts. This is a recorded development protocol, not a claim
about the private scorer's timeout. Automatic discovery of secret bundles is disabled.

Download `/kaggle/working/submission.zip` for the **candidate agent**, then upload it via
**Submit Prediction → File Upload**. The baseline ZIP and development records are for
comparison. Retain `selection.json`, `run-manifest.json` and the result directory to
reproduce the comparison. This notebook never submits automatically.
""")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"repair-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + "\n")
    print(f"Created {output}; embedded archives: " + ", ".join(f"{key}={value['sha256']}" for key, value in entries.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing candidate ZIP instead of rebuilding submission/.")
    parser.add_argument("--baseline", type=Path, help="Use an existing baseline ZIP instead of rebuilding the preserved source.")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/kaggle_evaluate.ipynb")
    args = parser.parse_args()
    archive = args.archive or ROOT / "dist/submission.zip"
    if args.archive is None:
        package_submission(ROOT / "submission", archive)
    baseline = None
    if not args.candidate_only:
        baseline = args.baseline or ROOT / "dist/submission-baseline-0.10.zip"
        if args.baseline is None:
            package_submission(ROOT / "experiments/baseline-010", baseline)
    build_notebook(archive, baseline, args.output)


if __name__ == "__main__":
    main()
