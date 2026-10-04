# Comparison baseline

`baseline-010/` preserves the exact source of the entry that the user reported
scored **0.10** on 2026-10-04. No per-task logs were available. Its deterministic
ZIP SHA-256 is:

```
c78c8f9868f37237b1869d5c1dc0777eaf8e42f145eee6311d3cba79333f2324
```

Rebuild it independently with:

```sh
python scripts/package_submission.py experiments/baseline-010 \
  --output dist/submission-baseline-0.10.zip
```

The candidate in `submission/` changes the reasoning budget from 1,024 to 2,048,
maximum output from 6,144 to 8,192, and command timeout from 45 to 90 seconds.
The five-minute session limit, sampling temperature and single-agent design remain
the same. Its prompt removes redundant startup work, encourages a first supported
patch early, and uses the official test runner's collection settings.

The target is **at least 0.13**, but no new model evaluation or leaderboard score
has been observed. Compilation, packaging checks and prompt review cannot measure
task resolution. Compare both archives on the same public development tasks using
`notebooks/kaggle_evaluate.ipynb`; retain task IDs, hashes, outcomes and durations.
An eight-task smoke run can expose broken execution or gross regressions; it
cannot reliably establish a three-percentage-point gain. Use a larger fixed public
set for selection and a fresh holdout before drawing performance conclusions.

Development traces and public reference patches must never be embedded in the
submission or made available to the inference agent. The public task verifier
uses its test specification only after inference.
