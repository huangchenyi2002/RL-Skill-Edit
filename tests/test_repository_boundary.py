import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rl_is_the_top_level_package():
    assert importlib.util.find_spec("rl_skill_edit") is not None
    assert not (ROOT / "baselines").exists()
