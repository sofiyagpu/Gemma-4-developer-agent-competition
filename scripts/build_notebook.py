#!/usr/bin/env python3
"""Build a self-contained Kaggle evaluation notebook from the packaged entry."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    archive = ROOT / "dist/submission.zip"
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    starter = json.loads((ROOT / "reference/official-getting-started.ipynb").read_text())
    cells = []

    def add(kind: str, source: str) -> None:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# Gemma 4: focused repository repair

This notebook reconstructs the exact submission archive, checks it with the official
harness, and optionally evaluates public development tasks. It does not submit to Kaggle.
No performance results are claimed until the evaluation cells have actually run.

Attach the **Gemma 4 Developer Agent competition**, the latest
**metric/gemma-4-developer-agent-wheelhouse** dataset, and
**google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2** model.
For evaluation choose **GPU L4 x4** and **Internet off**. Start a fresh session after
changing the wheelhouse. The default stops after CPU compilation; enable evaluation below.

Runtime setup and evaluation are adapted from [Ryan Holbrook's official starter,
version 2](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent).
The prompt and agent configuration in this archive are original to this entry.
""")
    add("code", """RUN_EVALUATION = False  # Set True to use the GPU and run real public tasks.
TASK_IDS = []  # Empty selects one deterministic task per repository (up to four).
""")
    add("markdown", "## Install the official offline runtime")
    add("code", "".join(starter["cells"][2]["source"]))
    add("markdown", "## Restore the exact entry and validate its hash")
    add("code", f'''import base64
import hashlib
import io
import json
import tempfile
import zipfile

ARCHIVE_SHA256 = {digest!r}
archive_bytes = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(archive_bytes).hexdigest() == ARCHIVE_SHA256
WORKING_DIR = Path('/kaggle/working')
WORKING_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = WORKING_DIR / 'submission.zip'
ZIP_PATH.write_bytes(archive_bytes)
AGENT_DIR = Path(tempfile.mkdtemp(prefix='repair-agent-', dir=WORKING_DIR))
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
    assert 'agent.yaml' in zf.namelist()
    for name in zf.namelist():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    zf.extractall(AGENT_DIR)
print('Submission:', ZIP_PATH, 'SHA-256:', ARCHIVE_SHA256)
''')
    add("code", """from adk_submission import ModelRegistry, compile_submission, validate_directory
from adk_submission.schema import SandboxedAgentConfig
from adk_submission.yaml_loader import load_yaml
from swegemma.config import build_submission_limits
from swegemma.models.discovery import validate_single_declared_model
from swegemma.tools import create_tools

limits, gen_constraints = build_submission_limits()
layout = validate_directory(AGENT_DIR, limits)
parsed = load_yaml(layout.config_path, layout.root_dir, limits=limits)
SandboxedAgentConfig.model_validate(parsed)
declared_model = validate_single_declared_model(AGENT_DIR)
compile_models = ModelRegistry()
compile_models.register(declared_model, declared_model)
compiled = compile_submission(
    submission_dir=AGENT_DIR, tool_registry=create_tools(ctx=None),
    model_registry=compile_models, limits=limits,
    generation_constraints=gen_constraints,
)
print('Official CPU compilation passed:', compiled.name, declared_model)
print('GPU evaluation enabled:', RUN_EVALUATION)
""")
    add("markdown", """## Optional real evaluation

This uses only the supplied public development tasks. The reference patches are available
to the verifier; they are not copied into the agent prompt or submission. A four-task
smoke run is diagnostic and does not estimate the hidden leaderboard reliably.
""")
    add("code", """if RUN_EVALUATION:
    from swegemma.models import load_tasks
    DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
    TASKS_PATH = DATA_DIR / 'tasks.jsonl'
    tasks = load_tasks(TASKS_PATH)
    GRAPH_DIR = str(DATA_DIR / 'graphs')
    EMBEDDINGS_DIR = str(DATA_DIR / 'embeddings')
    if TASK_IDS:
        by_id = {task.instance_id: task for task in tasks}
        unknown = set(TASK_IDS) - by_id.keys()
        if unknown:
            raise ValueError(f'Unknown task IDs: {sorted(unknown)}')
        selected_tasks = [by_id[key] for key in TASK_IDS]
    else:
        by_repo = {}
        for task in sorted(tasks, key=lambda task: task.instance_id):
            by_repo.setdefault(task.repo, task)
        selected_tasks = [by_repo[key] for key in sorted(by_repo)][:4]
    print('Selected:', [task.instance_id for task in selected_tasks])
""")
    server = "".join(starter["cells"][8]["source"])
    server = server.replace("gpu_memory_utilization=0.90", "gpu_memory_utilization=0.80")
    server = server.replace("gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1", "gpu_count = torch.cuda.device_count()\nassert gpu_count == 4, 'Select GPU L4 x4 for the competition model'")
    add("code", "server_instance = None\nif RUN_EVALUATION:\n" + "".join("    " + line + "\n" for line in server.splitlines()))
    evaluate = "".join(starter["cells"][10]["source"])
    evaluate = evaluate.replace("SAMPLE_TASKS = tasks[:2]", "SAMPLE_TASKS = selected_tasks")
    evaluate = evaluate.replace("compaction_interval=15", "compaction_interval=5")
    evaluate = evaluate.replace("predictions = []", "predictions = []\nmetrics = []")
    evaluate = evaluate.replace("    predictions.append(", "    metrics.append({'id': task.instance_id, 'repo': task.repo, 'resolved': result.resolved,\n                    'test_exit_code': result.test_exit_code, 'duration_seconds': result.duration_seconds,\n                    'tool_calls': result.tool_calls, 'patch_chars': len(result.agent_patch or '')})\n    predictions.append(")
    evaluate += """
metrics_df = pd.DataFrame(metrics)
display(metrics_df)
summary = {'archive_sha256': ARCHIVE_SHA256, 'task_count': len(metrics),
           'resolved_count': sum(bool(row['resolved']) for row in metrics),
           'resolution_rate': sum(bool(row['resolved']) for row in metrics) / len(metrics),
           'tasks': metrics}
(WORKING_DIR / 'development-summary.json').write_text(json.dumps(summary, indent=2))
submission_df.to_json(WORKING_DIR / 'development-patches.jsonl', orient='records', lines=True)
print(json.dumps(summary, indent=2))
"""
    add("code", "if RUN_EVALUATION:\n    try:\n" + "".join("        " + line + "\n" for line in evaluate.splitlines()) + "    finally:\n        if server_instance is not None:\n            server_instance.stop()\n")
    add("markdown", """## Submission artifact

Download `/kaggle/working/submission.zip`. It is the agent package, not the development
patches JSONL. Upload it to the competition's **Submit Prediction → File Upload** panel.
The local development summary is not a Kaggle score. Retain the SHA-256 when comparing runs.
""")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"repair-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
    output = ROOT / "notebooks/kaggle_evaluate.ipynb"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + "\n")
    print(f"Created {output} with archive SHA-256 {digest}")


if __name__ == "__main__":
    main()
