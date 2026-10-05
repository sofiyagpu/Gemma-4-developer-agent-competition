# Development

Run commands from the repository root. Local packaging needs Python 3.9+ and
`requirements-dev.txt`; official compilation uses a separate Python 3.12 environment.

## Build artifacts

```sh
python scripts/build_notebook.py
```

- `dist/submission.zip` — competition entry, with `agent.yaml` at the archive root.
- `dist/submission.manifest.json` — archive and source file SHA-256 hashes.
- `dist/submission-baseline-0.10.zip` — preserved comparison entry, rebuilt from its source.
- `notebooks/kaggle_evaluate.ipynb` — notebook embedding that exact archive.

This one command validates and rebuilds both ZIPs from source before embedding them
in the notebook, including on a clean checkout with no `dist/` directory. It avoids
embedding an older archive after changing `submission/`. Explicit `--archive` or
`--baseline` arguments use existing ZIPs and check their manifest hashes when present.
Commit the updated notebook with the source changes. The ZIP uses fixed timestamps, permissions and
entry ordering so identical source files produce identical bytes.

GitHub Actions runs package tests, validates the entry, rebuilds the notebook,
checks that it matches the committed version, and makes the build available as an artifact.
It does not run the model or submit to Kaggle.

## Agent design

One agent shares context across localization, editing, and testing. This avoids paying additional model turns for obligatory delegation on straightforward issues. Optional graph navigation uses real symbol names because the offline similarity tool resolves existing graph embeddings rather than embedding natural-language queries. Current source files take precedence over graph snippets.

The budget is five minutes per task, 80 counted calls, 100 turns, and 90 seconds per command. The previous entry used a 45-second command timeout. Increasing the command timeout allows a focused test more time to finish without increasing the whole session's time limit. At 120 tasks, five minutes per task would allow ten hours of agent time if evaluated serially; setup and other overhead still have to fit the competition's twelve-hour cap. The hidden task count and scoring concurrency have not been independently verified. This is a planning bound, not a measured runtime guarantee.

Generation uses temperature 0.2, top-p 0.95, 8,192 maximum output tokens, and a 2,048-token thinking budget. The previous entry used 6,144 and 1,024 tokens respectively. The official compiler forwards the reasoning budget to vLLM's `thinking_token_budget`; the output allowance also needs to leave room in the 32,768-token context. The shorter prompt starts with issue-specific inspection, makes a supported source edit early, and matches the official verifier's pytest collection options. These are unmeasured optimization hypotheses, not a claim of a higher score. LoRA adapters are not included.

## Official compilation and real evaluation

`scripts/validate_official.py` uses the actual competition compiler, schemas, generation constraints, and tool factories. It does not execute tools or call a model:

```sh
python3.12 -m venv .compile-venv
.compile-venv/bin/python -m pip install -r requirements-harness.txt
.compile-venv/bin/python scripts/validate_official.py --report dist/official-validation.json
.compile-venv/bin/python scripts/smoke_notebook.py --report dist/notebook-smoke.json
```

The exact official package sources are recorded in [source provenance](../reference/SOURCES.md). Install those packages and their dependencies in a Python 3.12 environment, or use the notebook with Kaggle's latest official wheelhouse. A successful CPU compile does not prove GPU serving or task resolution.

To run real public development tasks, import `notebooks/kaggle_evaluate.ipynb` into Kaggle, attach the competition dataset, the latest `metric/gemma-4-developer-agent-wheelhouse`, and Google's competition model version 2. Select **GPU L4 x4**, turn **Internet off**, and set `RUN_EVALUATION = True`. By default it selects eight seeded, repository-stratified public tasks and runs both baseline and candidate. Set `TASK_IDS` for a fixed comparison set, or `EXCLUDE_TASK_IDS` for a fresh holdout. Eight tasks are diagnostic, not enough to establish a three-percentage-point improvement.

Each archive keeps its own agent budgets. Public verification uses the same 300-second
command timeout for both variants so the candidate's longer agent command timeout does
not also buy it extra grading time. This is a recorded development setting, not a claim
about the private scorer. The runner uses public task specifications and snapshots;
automatic secret-bundle discovery is disabled. Missing public inputs are reported before
GPU startup instead of silently producing zero scores.

Results, patches, errors and trace paths are saved after each task under
`/kaggle/working/public-evaluation/<RUN_LABEL>/`. The run manifest records archive hashes,
task selection, package versions and evaluation settings. Reusing the same label resumes
completed tasks; changed inputs or a retry after a runtime exception require a new label.
`MAX_RUN_MINUTES` stops scheduling new task pairs; setup and in-flight work can run longer.

The notebook defaults to compilation only so an ordinary Run All does not silently consume GPU evaluation time. The CPU smoke script actually executes these cells with installed official packages in a temporary working directory. It skips Kaggle's wheel installer, does not start a model server, and does not measure patch quality. GPU execution is a separate validation step.

## Investigating a notebook exception

First distinguish a failure of the development notebook from a failure of Kaggle's
private scoring notebook after uploading the ZIP. They are different executions.
The generic **Notebook threw exception** status alone does not identify the cause.

For the development notebook, inspect the failing cell and
`/kaggle/working/runtime-error.txt` for server/evaluation errors; task-level unexpected
exceptions also retain full tracebacks in `records/*.json`. Check the attached inputs
and start a fresh session after updating the wheelhouse. For a ZIP scoring failure,
retain the uploaded archive's SHA-256 and the submission traceback or error details.
Local compilation cannot diagnose a private GPU/OOM/environment failure without that log.

## Submission

Upload `dist/submission.zip` through the competition's
[Submit Prediction](https://www.kaggle.com/competitions/gemma-4-developer-agent/submissions)
form. Track the score and errors there. Compilation and packaging checks do not
measure coding-agent performance.

## Repository layout

```text
submission/   Agent configuration and prompt
scripts/      Validation, packaging and notebook generation
notebooks/    Self-contained Kaggle evaluation notebook
tests/        Package-contract tests
docs/         Development instructions
reference/    Official guide, starter notebook and provenance
```

Official wheel packages are fetched by `requirements-harness.txt`, with hashes
recorded in the download URLs. They are not committed to the repository. The
reference notebook is retained because `scripts/build_notebook.py` reuses its
runtime setup and evaluation cells. See [source provenance](../reference/SOURCES.md).

The local validator supports this repository's single-agent configuration; it is
not a general ADK compiler. Use `scripts/validate_official.py` for official CPU
compilation, then the Kaggle notebook for real GPU evaluation.
