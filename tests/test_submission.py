"""Contract and archive tests; these do not execute Gemma or the official harness."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import yaml

from scripts.package_submission import package_submission
from scripts.validate_submission import MODEL, TOOLS, ValidationError, validate_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "submission"
        (self.root / "prompts").mkdir(parents=True)
        (self.root / "configs").mkdir()
        self.agent = {
            "agent_class": "LlmAgent", "name": "repair_agent", "model": MODEL,
            "instruction": "Follow the task and submit a verified patch.",
            "tools": sorted(TOOLS),
            "generate_content_config": {
                "temperature": 0.2, "top_p": 0.95, "max_output_tokens": 6144,
                "thinking_config": {"include_thoughts": True, "thinking_budget": 1024},
            },
        }
        self.write_agent()

    def write_agent(self):
        (self.root / "agent.yaml").write_text(yaml.safe_dump(self.agent), encoding="utf-8")

    def write_included_submission(self):
        text = yaml.safe_dump(self.agent)
        instruction_line = "instruction: Follow the task and submit a verified patch."
        text = text.replace(instruction_line, "instruction: !include prompts/system.md")
        generation = self.agent.pop("generate_content_config")
        text = yaml.safe_dump(self.agent).replace(instruction_line, "instruction: !include prompts/system.md")
        text += "generate_content_config: !include configs/generation.yaml\n"
        (self.root / "agent.yaml").write_text(text, encoding="utf-8")
        (self.root / "prompts/system.md").write_text("Read, repair, verify, submit.\n", encoding="utf-8")
        (self.root / "configs/generation.yaml").write_text(yaml.safe_dump(generation), encoding="utf-8")
        (self.root / "eval_config.yaml").write_text(
            "evaluation:\n  timeout_seconds: 45\n  max_tool_calls: 80\n  max_time_minutes: 5\n  max_turns: 100\n",
            encoding="utf-8",
        )

    def assert_invalid(self, pattern):
        with self.assertRaisesRegex(ValidationError, pattern):
            validate_submission(self.root)

    def test_inline_configuration(self):
        checked = validate_submission(self.root)
        self.assertEqual(checked.agent["model"], MODEL)
        self.assertEqual(checked.evaluation, {})

    def test_include_configuration_and_budget(self):
        self.write_included_submission()
        checked = validate_submission(self.root)
        self.assertEqual(len(checked.files), 4)
        self.assertEqual(checked.agent["instruction"], "Read, repair, verify, submit.\n")
        self.assertEqual(checked.agent["generate_content_config"]["max_output_tokens"], 6144)
        self.assertEqual(checked.evaluation["max_tool_calls"], 80)

    def test_archive_root_manifest_and_reproducibility(self):
        self.write_included_submission()
        output = Path(self.temporary.name) / "dist/submission.zip"
        first_manifest = package_submission(self.root, output)
        first_bytes = output.read_bytes()
        for path in self.root.rglob("*"):
            os.utime(path, (1_700_000_000, 1_700_000_000))
        second_manifest = package_submission(self.root, output)
        self.assertEqual(first_bytes, output.read_bytes())
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["archive"]["sha256"], hashlib.sha256(first_bytes).hexdigest())
        self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text()), first_manifest)
        with zipfile.ZipFile(output) as archive:
            expected = ["agent.yaml", "configs/generation.yaml", "eval_config.yaml", "prompts/system.md"]
            self.assertEqual(archive.namelist(), expected)
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(info.external_attr >> 16, 0o100644)
                self.assertEqual(archive.read(info.filename), (self.root / info.filename).read_bytes())
            self.assertIsNone(archive.testzip())

    def test_no_package_inside_source(self):
        with self.assertRaisesRegex(ValidationError, "outside"):
            package_submission(self.root, self.root / "submission.zip")
        self.assertFalse((self.root / "submission.zip").exists())

    def test_missing_agent(self):
        (self.root / "agent.yaml").unlink()
        self.assert_invalid("agent.yaml is missing")

    def test_reject_extra_root_and_python_files(self):
        for name in ("agent.yml", "README.md", "agent.py", "prompts/helper.py", ".env"):
            with self.subTest(name=name):
                path = self.root / name
                path.write_text("unexpected")
                self.assert_invalid("allowlist|Unsupported filename")
                path.unlink()

    def test_reject_unreferenced_prompt(self):
        (self.root / "prompts/unused.md").write_text("Not included")
        self.assert_invalid("Unreferenced")

    def test_reject_symlink_files_directories_and_root(self):
        target = Path(self.temporary.name) / "outside.md"
        target.write_text("Outside")
        link = self.root / "prompts/link.md"
        link.symlink_to(target)
        self.assert_invalid("Symlinks")
        link.unlink()
        # Even a symlink to an in-root target is deliberately excluded.
        link.symlink_to(self.root / "agent.yaml")
        self.assert_invalid("Symlinks")
        link.unlink()
        directory_link = self.root / "configs/link"
        directory_link.symlink_to(self.root / "prompts", target_is_directory=True)
        self.assert_invalid("Symlinks")
        directory_link.unlink()
        root_link = Path(self.temporary.name) / "root_link"
        root_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValidationError, "root must not be a symlink"):
            validate_submission(root_link)

    def test_reject_nonregular_file(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation unavailable")
        os.mkfifo(self.root / "prompts/pipe.txt")
        self.assert_invalid("Only regular files")

    def test_reject_unsafe_include_paths(self):
        for value in ("../outside.md", "/etc/passwd", "C:/private/file.md", "prompts/../agent.yaml", "prompts//system.md", "prompts\\system.md"):
            with self.subTest(value=value):
                self.write_agent()
                with (self.root / "agent.yaml").open("a") as handle:
                    handle.write("global_instruction: !include " + json.dumps(value) + "\n")
                self.assert_invalid("path|filename")

    def test_reject_missing_include(self):
        with (self.root / "agent.yaml").open("a") as handle:
            handle.write("global_instruction: !include prompts/missing.md\n")
        self.assert_invalid("does not exist")

    def test_reject_include_cycle(self):
        with (self.root / "agent.yaml").open("a") as handle:
            handle.write("global_instruction: !include configs/cycle.yaml\n")
        (self.root / "configs/cycle.yaml").write_text("!include cycle.yaml\n")
        self.assert_invalid("Include cycle")

    def test_reject_include_depth(self):
        with (self.root / "agent.yaml").open("a") as handle:
            handle.write("global_instruction: !include configs/chain0.yaml\n")
        for index in range(12):
            (self.root / f"configs/chain{index}.yaml").write_text(
                f"!include chain{index + 1}.yaml\n" if index < 11 else "done\n"
            )
        self.assert_invalid("Include depth")

    def test_nested_include_uses_including_directory(self):
        self.write_included_submission()
        generation = (self.root / "configs/generation.yaml").read_text()
        (self.root / "configs/generation.yaml").write_text("!include actual.yaml\n")
        (self.root / "configs/actual.yaml").write_text(generation)
        checked = validate_submission(self.root)
        self.assertEqual(checked.agent["generate_content_config"]["max_output_tokens"], 6144)

    def test_reject_duplicate_yaml_keys(self):
        with (self.root / "agent.yaml").open("a") as handle:
            handle.write("model: bad-model\n")
        self.assert_invalid("Duplicate YAML key")

    def test_reject_yaml_alias_anchor_and_python_tags(self):
        for value in ("&anchor value", "*missing", "!!python/object/apply:os.system [echo forbidden]"):
            with self.subTest(value=value):
                self.write_agent()
                with (self.root / "agent.yaml").open("a") as handle:
                    handle.write(f"description: {value}\n")
                self.assert_invalid("anchors and aliases|Invalid YAML")

    def test_reject_deep_yaml_nesting(self):
        with (self.root / "agent.yaml").open("a") as handle:
            handle.write("description: " + "[" * 55 + "0" + "]" * 55 + "\n")
        self.assert_invalid("nesting exceeds")

    def test_reject_nonmapping_yaml(self):
        (self.root / "agent.yaml").write_text("- hello\n")
        self.assert_invalid("must be a mapping")

    def test_reject_non_utf8_and_nul(self):
        for content, error in ((b"\xff", "UTF-8"), (b"name: '\x00'", "NUL")):
            with self.subTest(content=content):
                (self.root / "agent.yaml").write_bytes(content)
                self.assert_invalid(error)

    def test_reject_wrong_model_agent_and_unknown_fields(self):
        for key, value, error in (
            ("model", "gemma-4-9b-it", "Competition model"),
            ("agent_class", "LoopAgent", "LlmAgent"),
            ("name", "invalid-name", "identifier"),
            ("instruction", "", "nonempty"),
            ("skills", ["skills/helper"], "Unsupported fields"),
            ("sub_agents", [], "Unsupported fields"),
        ):
            with self.subTest(key=key):
                old = dict(self.agent)
                self.agent[key] = value
                self.write_agent()
                self.assert_invalid(error)
                self.agent = old

    def test_reject_missing_duplicate_unknown_and_structured_tools(self):
        for tools in (sorted(TOOLS)[:-1], sorted(TOOLS) + ["run_command"], ["network"] * 9,
                      [{"agent_tool": {"config_path": "other.yaml"}}]):
            with self.subTest(tools=tools):
                self.agent["tools"] = tools
                self.write_agent()
                self.assert_invalid("tools|tool names")

    def test_reject_invalid_generation(self):
        for config in (
            {"temperature": -1}, {"top_p": 1.1}, {"top_p": float("nan")},
            {"temperature": float("inf")}, {"max_output_tokens": 32769},
            {"max_output_tokens": True}, {"max_output_tokens": 0},
            {"top_k": 1.5}, {"seed": "0"}, {"tools": []},
            {"thinking_config": {"thinking_budget": -1}},
            {"thinking_config": {"thinking_budget": 32769}},
            {"thinking_config": {"include_thoughts": "true"}},
            {"thinking_config": {"thinking_level": "huge"}},
            {"stop_sequences": "STOP"},
        ):
            with self.subTest(config=config):
                self.agent["generate_content_config"] = config
                self.write_agent()
                self.assert_invalid("must|Unsupported")

    def test_reject_invalid_budgets(self):
        for content in (
            {"evaluation": {"max_tool_calls": 0}}, {"evaluation": {"timeout_seconds": -1}},
            {"evaluation": {"max_turns": True}}, {"evaluation": {"max_time_minutes": float("nan")}},
            {"evaluation": {"max_time_minutes": "5"}}, {"evaluation": {"timeout": 60}},
            {"max_tool_calls": 80}, {"evaluation": None},
        ):
            with self.subTest(content=content):
                (self.root / "eval_config.yaml").write_text(yaml.safe_dump(content))
                self.assert_invalid("must|Unsupported|mapping")

    def test_validation_cli_failure_is_clear(self):
        self.agent["model"] = "wrong"
        self.write_agent()
        script = Path(__file__).resolve().parents[1] / "scripts/validate_submission.py"
        result = subprocess.run([sys.executable, str(script), str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Validation failed: Competition model", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
