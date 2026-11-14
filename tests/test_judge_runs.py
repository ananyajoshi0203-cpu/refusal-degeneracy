import importlib.util
import sys
from pathlib import Path

import pytest

from refusal_degeneracy.runs import save_run

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "judge_runs.py"
MODEL = "org/Model-7B"
UNSAFE_MARK = "UNSAFE"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("judge_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def flag_marked(prompts, replies):
    return [UNSAFE_MARK in reply for reply in replies]


@pytest.mark.parametrize(
    ("replies", "expected_asr", "expected_refusal"),
    [
        ([f"{UNSAFE_MARK} sure", "I cannot", "I'm sorry"], 1 / 3, 2 / 3),
        ([f"{UNSAFE_MARK} here", f"{UNSAFE_MARK} ok"], 1.0, 0.0),
    ],
)
def test_each_reply_file_gets_its_asr_and_keyword_refusal_rate(
    script, monkeypatch, tmp_path, replies, expected_asr, expected_refusal
):
    monkeypatch.setenv("RD_RESULTS", str(tmp_path))
    prompts = [f"p{i}" for i in range(len(replies))]
    save_run(MODEL, "test_jbb__direction_ablation", prompts, replies, "test_jbb")
    save_run(MODEL, "test_jbb__baseline", prompts, ["I cannot"] * len(prompts), "test_jbb")

    verdicts = script.judge_files(script.reply_files(tmp_path), flag_marked)

    by_run = {(v.test_set, v.intervention): v for v in verdicts}
    attacked = by_run[("test_jbb", "direction_ablation")]
    assert (attacked.n, attacked.asr, attacked.keyword_refusal_rate) == (
        len(replies), pytest.approx(expected_asr), pytest.approx(expected_refusal)
    )
    assert by_run[("test_jbb", "baseline")].asr == 0.0
