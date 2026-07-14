from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TRACKED_PATHS = (
    "baselines",
    "config.yaml",
    "data/mock_rl_skill_edit/current_events.jsonl",
    "data/mock_rl_skill_edit/current_history.json",
    "data/mock_rl_skill_edit/current_skill.md",
    "data/spreadsheet",
    "docs/superpowers/plans/2026-07-15-rl-skill-edit.md",
    "docs/superpowers/specs/2026-07-15-rl-skill-edit-design.md",
    "experiments",
    "rl_skill_edit/random_policy.py",
    "src",
    "study1_main.py",
)

PUBLIC_DOCUMENTS = (
    "README.md",
    "ARCHITECTURE.md",
    "CONTEXT.md",
    "configs/rl_skill_edit.yaml",
    "configs/rl_skill_edit_smoke.yaml",
    "data/initial_skill.md",
    "docs/rl_skill_edit_implementation_note.md",
    "scripts/run_smoke.sh",
)

FORBIDDEN_PUBLIC_PATTERNS = (
    r"\bcurrent_method\b",
    r"\brandom_edit(?:_search)?\b",
    r"\brandom_policy\b",
    r"\bteacher\b",
    r"\breference\b",
    r"\bosd\b",
    r"\bstudy1_main\.py\b",
    r"\brequirements-rl\.txt\b",
    r"\bdata/spreadsheet/",
    r"\bsrc/",
    r"\bbaselines/",
    r"\bexperiments/",
)

README_OPENING = """# RL-Skill-Edit

RL-Skill-Edit optimizes one external Markdown Skill while the agent model remains frozen.
The actor-critic selects a Skill module and edit operator; an Editor proposes one local
patch; Train reward updates the policy; Validation selects the checkpoint; Test is read
only after the final Skill is frozen.
"""


def _tracked_files() -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(path.decode("utf-8") for path in completed.stdout.split(b"\0") if path)


def _matches_path(path: str, forbidden: str) -> bool:
    return path == forbidden or path.startswith(f"{forbidden}/")


def test_rl_is_the_top_level_package() -> None:
    assert importlib.util.find_spec("rl_skill_edit") is not None


def test_published_tree_contains_no_original_method_paths() -> None:
    tracked = _tracked_files()
    forbidden = [
        path
        for path in tracked
        if any(_matches_path(path, prefix) for prefix in FORBIDDEN_TRACKED_PATHS)
    ]
    assert forbidden == []

    runtime_python = [
        path
        for path in tracked
        if path.endswith(".py") and not path.startswith(("rl_skill_edit/", "tests/"))
    ]
    assert runtime_python == []


def test_runtime_has_no_original_namespace_imports_or_method_branches() -> None:
    forbidden_import_roots = {"baselines", "experiments", "src"}
    invalid_imports: list[str] = []
    invalid_markers: list[str] = []
    for path in sorted((ROOT / "rl_skill_edit").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.split(".", 1)[0] in forbidden_import_roots:
                    invalid_imports.append(f"{path.relative_to(ROOT)}:{module}")
        for marker in (
            r"\bcurrent_method\b",
            r"\brandom_edit(?:_search)?\b",
            r"\brandom_policy\b",
            r"\bteacher\b",
            r"\bosd\b",
            r"\breference_(?:baseline|endpoint|method|rollout)\b",
        ):
            if re.search(marker, source, flags=re.IGNORECASE):
                invalid_markers.append(f"{path.relative_to(ROOT)}:{marker}")

    assert invalid_imports == []
    assert invalid_markers == []


def test_public_documents_describe_only_rl_skill_edit() -> None:
    violations: list[str] = []
    for relative in PUBLIC_DOCUMENTS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PUBLIC_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                violations.append(f"{relative}:{pattern}")

    assert violations == []

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith(README_OPENING)
    assert "python -m rl_skill_edit --config configs/rl_skill_edit.yaml" in readme
    assert "--test-only" in readme
    assert "scripts/run_smoke.sh" in readme


def test_real_config_uses_the_rl_only_initial_skill() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/rl_skill_edit.yaml").read_text(encoding="utf-8")
    )
    assert config["paths"]["initial_skill"] == "data/initial_skill.md"

    initial_skill = ROOT / "data/initial_skill.md"
    assert initial_skill.is_file()
    text = initial_skill.read_text(encoding="utf-8")
    for pattern in FORBIDDEN_PUBLIC_PATTERNS:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_dependency_file_contains_only_standalone_runtime_and_test_tools() -> None:
    packages = {
        line.split("==", 1)[0].casefold()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert packages == {
        "httpx",
        "numpy",
        "openpyxl",
        "pytest",
        "pyyaml",
        "ruff",
    }


def test_private_real_run_inputs_are_not_publishable_by_default() -> None:
    ignored = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "data/private/" in ignored


def test_cli_has_no_method_selector() -> None:
    help_text = subprocess.run(
        [sys.executable, "-m", "rl_skill_edit", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "--methods" not in help_text
