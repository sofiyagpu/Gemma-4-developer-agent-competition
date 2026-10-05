"""Regression checks for missing/stale artifacts and the paired evaluation driver."""
from __future__ import annotations

import ast
import asyncio
import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts import build_notebook
from scripts.package_submission import package_submission
from scripts.public_eval import public_evaluator_class, select_tasks, summarize

ROOT = Path(__file__).resolve().parents[1]


class NotebookBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("submission", "experiments", "reference", "scripts"):
            shutil.copytree(ROOT / directory, self.root / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        self.output = self.root / "notebooks/kaggle_evaluate.ipynb"

    def build(self):
        with patch.object(build_notebook, "ROOT", self.root), patch("sys.argv", ["build_notebook"]):
            build_notebook.main()

    def test_clean_checkout_builds_both_archives_and_matching_notebook(self):
        self.assertFalse((self.root / "dist").exists())
        self.build()
        notebook = json.loads(self.output.read_text())
        entries = None
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"cell-{index}", "exec")
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARCHIVES" for t in node.targets):
                    entries = ast.literal_eval(node.value)
        self.assertEqual(set(entries), {"baseline", "candidate"})
        for name, filename, directory in (
            ("candidate", "submission.zip", "submission"),
            ("baseline", "submission-baseline-0.10.zip", "experiments/baseline-010"),
        ):
            archive = self.root / "dist" / filename
            payload = archive.read_bytes()
            self.assertEqual(base64.b64decode(entries[name]["base64"]), payload)
            self.assertEqual(entries[name]["sha256"], hashlib.sha256(payload).hexdigest())
            with zipfile.ZipFile(archive) as zf:
                for path in zf.namelist():
                    self.assertEqual(zf.read(path), (self.root / directory / path).read_bytes())
        self.assertNotEqual(entries["baseline"]["sha256"], entries["candidate"]["sha256"])

    def test_rebuild_replaces_stale_archive_and_is_reproducible(self):
        self.build()
        old_bytes = self.output.read_bytes()
        target = self.root / "dist/submission.zip"
        shutil.copyfile(self.root / "dist/submission-baseline-0.10.zip", target)
        self.build()
        self.assertEqual(old_bytes, self.output.read_bytes())
        prompt = self.root / "submission/prompts/system.md"
        prompt.write_text(prompt.read_text() + "\nVerify the requested behavior.\n")
        self.build()
        self.assertNotEqual(old_bytes, self.output.read_bytes())

    def test_explicit_archive_rejects_mismatched_manifest(self):
        archive = self.root / "dist/custom.zip"
        package_submission(self.root / "submission", archive)
        archive.write_bytes(b"stale or damaged archive")
        with self.assertRaisesRegex(ValueError, "manifest"):
            build_notebook.build_notebook(archive, None, self.output)


class EvaluationTests(unittest.TestCase):
    def test_sampling_is_order_independent_and_holdout_disjoint(self):
        tasks = [SimpleNamespace(instance_id=f"{repo}-{n}", repo=repo)
                 for repo in ("large", "small") for n in range(5 if repo == "large" else 2)]
        panel = select_tasks(tasks, count=4)
        self.assertEqual(panel, select_tasks(list(reversed(tasks)), count=4))
        self.assertEqual({t.repo for t in panel}, {"large", "small"})
        used = [t.instance_id for t in panel]
        holdout = select_tasks(tasks, count=4, exclude_ids=used)
        self.assertFalse(set(used) & {t.instance_id for t in holdout})
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_tasks(tasks, task_ids=used, exclude_ids=used)

    def test_comparison_uses_only_complete_pairs(self):
        def row(variant, task, resolved):
            return dict(variant=variant, id=task, repo="repo", resolved=resolved,
                        failure_category="resolved" if resolved else "tests_failed")
        rows = [row("baseline", "a", False), row("candidate", "a", True),
                row("baseline", "b", True)]
        result = summarize(rows, ["a", "b"], ["baseline", "candidate"])
        self.assertEqual(result["paired"]["completed_pairs"], 1)
        self.assertEqual(result["paired"]["wins"], ["a"])
        self.assertEqual(result["paired"]["regressions"], [])

    def test_equal_grading_timeout_and_restore_after_exception(self):
        @dataclass
        class Config:
            timeout_seconds: int

        class Base:
            def __init__(self, config):
                self.config = config
                self.docker = SimpleNamespace(timeout_seconds=config.timeout_seconds)

            async def _run_agent_sandbox(self):
                self.agent_timeout = self.config.timeout_seconds
                return "patch"

            async def evaluate_task(self, fail=False):
                await self._run_agent_sandbox()
                self.grading_timeout = self.config.timeout_seconds
                self.grading_default = self.docker.timeout_seconds
                if fail:
                    raise RuntimeError("grading failure")
                return "result"

        for timeout in (45, 90):
            config = Config(timeout)
            evaluator = public_evaluator_class(Base)(config, 300)
            self.assertEqual(asyncio.run(evaluator.evaluate_task()), "result")
            self.assertEqual(evaluator.agent_timeout, timeout)
            self.assertEqual(evaluator.grading_timeout, 300)
            self.assertEqual(evaluator.grading_default, 300)
            self.assertIs(evaluator.config, config)
            with self.assertRaisesRegex(RuntimeError, "grading failure"):
                asyncio.run(evaluator.evaluate_task(fail=True))
            self.assertIs(evaluator.config, config)
            self.assertEqual(evaluator.docker.timeout_seconds, timeout)
            self.assertEqual(evaluator._get_secret_bundle_data(), ({}, {}, None))


if __name__ == "__main__":
    unittest.main()
