# Gemma 4 Developer Agent

A coding agent for the [Kaggle competition](https://www.kaggle.com/competitions/gemma-4-developer-agent).
Uses Gemma 4 to navigate repositories, fix issues and verify changes with targeted tests.

## Quick start

Python 3.9+.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/package_submission.py
```

Upload `dist/submission.zip` to Kaggle.

The previous entry with a user-reported score of **0.10** is preserved in
[`experiments/baseline-010/`](experiments/baseline-010/). The current candidate
has a shorter repair prompt, a larger reasoning budget and longer command timeout.
**Its leaderboard score has not been measured.** Use the evaluation notebook to
compare both entries on the same public tasks before concluding it improved.

## Development

```sh
python -m unittest discover -s tests -v
python scripts/build_notebook.py
```

Agent settings live in [`submission/`](submission/).
Use the [evaluation notebook](notebooks/kaggle_evaluate.ipynb) to run public tasks on Kaggle.

[Development guide](docs/development.md) · [Official sources](reference/SOURCES.md)
