from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

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


def _casefold_path_components(path: str) -> tuple[str, ...]:
    return tuple(component.casefold() for component in PurePosixPath(path).parts)


def _matches_path(path: str, forbidden: str) -> bool:
    path_components = _casefold_path_components(path)
    forbidden_components = _casefold_path_components(forbidden)
    return bool(forbidden_components) and (
        path_components[: len(forbidden_components)] == forbidden_components
    )


def _is_unexpected_runtime_python_path(path: str) -> bool:
    components = _casefold_path_components(path)
    return (
        bool(components)
        and components[-1].endswith(".py")
        and components[0]
        not in {
            "rl_skill_edit",
            "tests",
        }
    )


def _original_namespace_imports(source: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    forbidden_roots = {"baselines", "experiments", "src"}
    invalid: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for module in modules:
            if module.split(".", 1)[0].casefold() in forbidden_roots:
                invalid.append(module)
    return tuple(invalid)


def test_rl_is_the_top_level_package() -> None:
    assert importlib.util.find_spec("rl_skill_edit") is not None


def test_boundary_helpers_reject_mixed_case_original_paths_and_imports() -> None:
    assert _matches_path("SRC/teacher.md", "src")
    assert _matches_path("Baselines/legacy.json", "baselines")
    assert _matches_path("src/agent.py", "SRC")
    assert not _matches_path("src_backup/agent.py", "src")
    assert _is_unexpected_runtime_python_path("SRC/agent.PY")
    assert _is_unexpected_runtime_python_path("EXPERIMENTS/run.py")
    assert not _is_unexpected_runtime_python_path("RL_SKILL_EDIT/cli.PY")
    assert not _is_unexpected_runtime_python_path("Tests/test_cli.py")
    assert _original_namespace_imports(
        "import Src.agent\nfrom EXPERIMENTS.x import runner\n"
    ) == ("Src.agent", "EXPERIMENTS.x")


def test_published_tree_contains_no_original_method_paths() -> None:
    tracked = _tracked_files()
    forbidden = [
        path
        for path in tracked
        if any(_matches_path(path, prefix) for prefix in FORBIDDEN_TRACKED_PATHS)
    ]
    assert forbidden == []

    runtime_python = [
        path for path in tracked if _is_unexpected_runtime_python_path(path)
    ]
    assert runtime_python == []


def test_runtime_has_no_original_namespace_imports_or_method_branches() -> None:
    invalid_imports: list[str] = []
    invalid_markers: list[str] = []
    for path in sorted((ROOT / "rl_skill_edit").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for module in _original_namespace_imports(source):
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
    for published_path in (
        "frozen_method_artifacts.json",
        "task_level_scores.csv",
        "comparison_rollout_cache.json",
        "rl_skill_edit/rollout_cache.json",
        "rl_skill_edit/editor_cache.json",
        "rl_skill_edit/episodes/episode_*/",
    ):
        assert f"`{published_path}`" in readme


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
