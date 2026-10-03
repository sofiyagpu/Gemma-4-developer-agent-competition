# Gemma 4 Developer Agent entry

A declarative coding agent for the [Gemma 4 Developer Agent competition](https://www.kaggle.com/competitions/gemma-4-developer-agent). The entry runs the required `gemma-4-31b-it-qat-w4a16-ct` model with the nine official tools. It searches the repository, identifies a causal fix, edits production code, runs focused checks, and submits the working-tree patch.

**Status:** the official CPU compiler check and all 24 local package tests passed. A reproducible entry and evaluation notebook have been prepared. See `dist/official-validation.json` for the compiler report and archive hash. These checks do not measure repair quality. No trained adapter or measured leaderboard score is claimed.

## Build and check

Python 3.9+ suffices for the local package tools. The official runtime requires a newer Python; use Python 3.12 for that environment.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_submission.py
.venv/bin/python scripts/package_submission.py
.venv/bin/python scripts/build_notebook.py
```

Outputs:

- `dist/submission.zip`: upload this file to Kaggle; `agent.yaml` is at its root.
- `dist/submission.manifest.json`: SHA-256 of the archive and each source file, stored outside the ZIP.
- `notebooks/kaggle_evaluate.ipynb`: self-contained notebook embedding the exact ZIP and checking its hash.

The ZIP contains only `submission/` files. It excludes development tools, credentials, virtual environments, reference notebooks, datasets, and validation reports. The build uses fixed timestamps, sorted entries, fixed permissions, and uncompressed ZIP entries to produce identical bytes across builds.

## Agent design

One agent shares context across localization, editing, and testing. This avoids paying additional model turns for obligatory delegation on straightforward issues. Optional graph navigation uses real symbol names because the offline similarity tool resolves existing graph embeddings rather than embedding natural-language queries. Current source files take precedence over graph snippets.

The initial budget is five minutes per task, 80 counted calls, 100 turns, and 45 seconds per command. With roughly 120 hidden tasks, five minutes per task allows ten hours of agent time if evaluated serially; setup and other overhead still have to fit the competition's twelve-hour cap. This is an initial setting, not a measured runtime guarantee. The prompt reserves time for diff review and explicit submission.

Generation uses temperature 0.2, top-p 0.95, 6,144 maximum output tokens, and a 1,024-token thinking budget. LoRA adapters are optional under the competition rules and are not included here. These settings need empirical comparison before any performance claim.

## Official compilation and real evaluation

`scripts/validate_official.py` uses the actual competition compiler, schemas, generation constraints, and tool factories. It does not execute tools or call a model:

```sh
python3.12 -m venv .compile-venv
.compile-venv/bin/python -m pip install -r requirements-harness.txt
.compile-venv/bin/python scripts/validate_official.py --report dist/official-validation.json
```

The exact official package sources are recorded in [reference/SOURCES.md](reference/SOURCES.md). Install those packages and their dependencies in a Python 3.12 environment, or use the notebook with Kaggle's latest official wheelhouse. A successful CPU compile does not prove GPU serving or task resolution.

To run real public development tasks, import `notebooks/kaggle_evaluate.ipynb` into Kaggle, attach the competition dataset, the latest `metric/gemma-4-developer-agent-wheelhouse`, and Google's competition model version 2. Select **GPU L4 x4**, turn **Internet off**, and set `RUN_EVALUATION = True`. By default it selects one deterministic task from each of up to four repositories. Set `TASK_IDS` for a fixed comparison set. It writes `development-summary.json`, development patches, and the official evaluator's results. Four tasks are a smoke test, not a reliable hidden-score estimate.

The notebook defaults to compilation only so an ordinary Run All does not silently consume GPU evaluation time. Its cells have been syntax-checked locally; GPU execution is a separate validation step.

## Submit

On [Kaggle's submissions page](https://www.kaggle.com/competitions/gemma-4-developer-agent/submissions), choose **Submit Prediction → File Upload**, upload `dist/submission.zip`, add a description, and submit. Inspect the resulting status and score before comparing or selecting a final entry. The UI showed one submission per day when checked on October 3, 2026.

The current final deadline is December 2, 2026 at 23:59 UTC (December 3 at 04:59 in Almaty). The separate paper track is optional and is not part of this entry. See the [official overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview) for current rules and dates.

## Files

- `submission/`: original agent configuration, generation settings, budget, and prompt.
- `scripts/`: local validation, official compilation, deterministic packaging, and notebook generation.
- `tests/`: package-contract tests, including unsafe paths, include resolution, and ZIP reproducibility.
- `reference/`: downloaded official harness guide, starter notebook, small official wheels, and provenance.

The local validator deliberately supports this repository's single-agent subset; it is not a replacement for the official compiler or a general implementation of every permitted ADK configuration.
