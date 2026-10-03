# Development

Run commands from the repository root. Local packaging needs Python 3.9+ and
`requirements-dev.txt`; official compilation uses a separate Python 3.12 environment.

## Build artifacts

```sh
python scripts/validate_submission.py
python scripts/package_submission.py
python scripts/build_notebook.py
```

- `dist/submission.zip` — competition entry, with `agent.yaml` at the archive root.
- `dist/submission.manifest.json` — archive and source file SHA-256 hashes.
- `notebooks/kaggle_evaluate.ipynb` — notebook embedding that exact archive.

After changing `submission/`, rebuild both the ZIP and notebook. Commit the updated
notebook with the source changes. The ZIP uses fixed timestamps, permissions and
entry ordering so identical source files produce identical bytes.

GitHub Actions runs package tests, validates the entry, rebuilds the notebook,
checks that it matches the committed version, and makes the build available as an artifact.
It does not run the model or submit to Kaggle.

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

The exact official package sources are recorded in [source provenance](../reference/SOURCES.md). Install those packages and their dependencies in a Python 3.12 environment, or use the notebook with Kaggle's latest official wheelhouse. A successful CPU compile does not prove GPU serving or task resolution.

To run real public development tasks, import `notebooks/kaggle_evaluate.ipynb` into Kaggle, attach the competition dataset, the latest `metric/gemma-4-developer-agent-wheelhouse`, and Google's competition model version 2. Select **GPU L4 x4**, turn **Internet off**, and set `RUN_EVALUATION = True`. By default it selects one deterministic task from each of up to four repositories. Set `TASK_IDS` for a fixed comparison set. It writes `development-summary.json`, development patches, and the official evaluator's results. Four tasks are a smoke test, not a reliable hidden-score estimate.

The notebook defaults to compilation only so an ordinary Run All does not silently consume GPU evaluation time. Its cells have been syntax-checked locally; GPU execution is a separate validation step.

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
