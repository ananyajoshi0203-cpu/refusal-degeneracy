import pytest
import torch

from refusal_degeneracy.runs import (
    checkpointed,
    completed_replies,
    load_run,
    results_root,
    run_path,
    save_run,
)


def test_results_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RD_RESULTS", str(tmp_path))
    assert results_root() == tmp_path


def test_save_and_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("RD_RESULTS", str(tmp_path))
    path = save_run("org/Model-1B", "baseline__test", ["p1", "p2"], ["r1", "r2"], "test", {"attack": "none"})
    rows = load_run(path)
    assert [r["prompt"] for r in rows] == ["p1", "p2"]
    assert rows[0]["meta"]["attack"] == "none"
    assert path == run_path("org/Model-1B", "baseline__test")


SAVED_PROMPTS = ["p1", "p2", "p3"]
SAVED_REPLIES = ["r1", "r2", "r3"]


@pytest.mark.parametrize(
    ("saved", "requested", "expected"),
    [
        (True, SAVED_PROMPTS, SAVED_REPLIES),
        (True, SAVED_PROMPTS[:2], None),
        (True, [*SAVED_PROMPTS, "p4"], None),
        (False, SAVED_PROMPTS, None),
    ],
    ids=["same-prompts", "different-limit", "cut-short", "never-saved"],
)
def test_completed_replies_reuses_only_a_run_over_the_same_prompts(
    monkeypatch, tmp_path, saved, requested, expected
):
    monkeypatch.setenv("RD_RESULTS", str(tmp_path))
    if saved:
        save_run("org/Model-1B", "test__baseline", SAVED_PROMPTS, SAVED_REPLIES, "test")

    replies = completed_replies("org/Model-1B", "test__baseline", requested)

    assert replies == expected


@pytest.mark.parametrize(("resume", "expected"), [(True, "saved"), (False, "computed")])
def test_checkpointed_loads_only_when_resuming(tmp_path, resume, expected):
    path = tmp_path / "stage.pt"
    torch.save("saved", path)

    value = checkpointed(path, resume, lambda: "computed")

    assert value == expected
    assert torch.load(path, weights_only=False) == expected
