# Official source provenance

Retrieved 2026-10-03 from the public Kaggle API. These are development references, excluded from the submission archive.

## Official starter notebook

- Source: https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent
- API: https://www.kaggle.com/api/v1/kernels/pull/ryanholbrook/getting-started-gemma-4-developer-agent
- Author: Ryan Holbrook (Kaggle staff). Version 2, last run 2026-09-30.
- `official-getting-started.ipynb` is the original notebook source returned by the API.
- SHA-256: `0c54c3bac3269e422b1d48ac8087ff26b49fe764231b7983c90ef842382f9a87`

## Official evaluation wheels

- Dataset: https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse
- Public listing API: https://www.kaggle.com/api/v1/datasets/list/metric/gemma-4-developer-agent-wheelhouse
- File URL pattern: `https://www.kaggle.com/api/v1/datasets/download/metric/gemma-4-developer-agent-wheelhouse/<filename>`
- The dataset is updated in place; these hashes identify the exact downloaded builds.

- `adk_eval_core-0.1.0-py3-none-any.whl`: SHA-256 `194dd8f9aab154857bfa0db9d9ec382db856ee124529e072509634d3071d8643`
- `adk_submission-0.2.12-py3-none-any.whl`: SHA-256 `077c438c426e625b9f722081694e1d32856e6f7e932ef625002fc4a11aabdc10`
- `swegemma-0.2.7-py3-none-any.whl`: SHA-256 `27a2f60f8db46c8fef5defc16df722dac0402446c9a6252e7e6b4c280e843c81`

Only the small CPU-compatible harness wheels are stored here. GPU libraries and Gemma weights are not included.

## Validation boundary

`scripts/validate_official.py` uses the real official compiler, schema, competition limits and tool factories. It compiles the declared agent tree without invoking tools or calling a model. No tool outcomes, inference, patch quality, GPU compatibility or Kaggle score are simulated. Those require evaluation on Kaggle with the official notebook/runtime.
