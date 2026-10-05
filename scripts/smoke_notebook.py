#!/usr/bin/env python3
"""Execute notebook CPU cells with installed official packages; no GPU or scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def smoke_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text())
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    previous_directory = os.environ.get("GEMMA_WORKING_DIR")
    executed, skipped = [], []
    state = {"__name__": "notebook_cpu_smoke"}
    try:
        with tempfile.TemporaryDirectory(prefix="gemma-notebook-check-") as directory:
            os.environ["GEMMA_WORKING_DIR"] = directory
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                if "kaggle-runtime-install" in cell.get("metadata", {}).get("tags", []):
                    skipped.append(index)
                    continue
                if "RUN_EVALUATION =" in source:
                    # Only the known, default compile-only notebook is allowed.
                    if "RUN_EVALUATION = False" not in source:
                        raise ValueError("CPU smoke requires RUN_EVALUATION = False")
                print(f"Executing CPU cell {index}", flush=True)
                exec(compile(source, f"{path.name}:cell-{index}", "exec"), state)
                if state.get("RUN_EVALUATION") is not False:
                    raise ValueError("Notebook must keep GPU evaluation disabled during CPU smoke")
                executed.append(index)
    finally:
        if previous_directory is None:
            os.environ.pop("GEMMA_WORKING_DIR", None)
        else:
            os.environ["GEMMA_WORKING_DIR"] = previous_directory
    return {"status": "passed", "executed_cells": executed, "skipped_install_cells": skipped,
            "scope": "Real CPU cell execution with installed official packages; "
                     "Kaggle wheel installation, GPU serving and task scoring were not run."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "notebooks/kaggle_evaluate.ipynb")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = smoke_notebook(args.notebook)
    report = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
