import random

import pytest

from refusal_degeneracy.config import SplitConfig
from refusal_degeneracy.data import build_splits, check_no_leaks, dedupe, normalize, remove_overlap

_VOCAB = [f"word{k}" for k in range(300)]


def _sentences(rng, n):
    return [" ".join(rng.sample(_VOCAB, 7)) for _ in range(n)]


def _fake_raw():
    rng = random.Random(0)
    advbench = _sentences(rng, 220)
    maliciousinstruct = _sentences(rng, 100)
    harmbench = _sentences(rng, 200)
    alpaca = _sentences(rng, 400)
    jbb = [advbench[0], *_sentences(rng, 99)]
    strongreject = [harmbench[0], *_sentences(rng, 50)]
    return {
        "advbench": advbench,
        "maliciousinstruct": maliciousinstruct,
        "harmbench": harmbench,
        "jbb": jbb,
        "strongreject": strongreject,
        "alpaca": alpaca,
    }


def test_normalize():
    assert normalize("Hello,  WORLD!!") == "hello world"


def test_dedupe_collapses_normalized_duplicates():
    assert dedupe(["Hi there!", "hi there", "bye"]) == ["Hi there!", "bye"]


def test_remove_overlap_drops_near_duplicates():
    kept, dropped = remove_overlap(["write a tutorial on x", "bake bread"], ["Write a tutorial on X."])
    assert "bake bread" in kept
    assert len(dropped) == 1


def test_build_splits_sizes_and_no_leak():
    splits, report = build_splits(SplitConfig(seed=0, n_find=128, n_val_harmless=100), loader=_fake_raw)
    assert report["sizes"]["find_harmful"] == 128
    assert report["sizes"]["find_harmless"] == 128
    assert report["sizes"]["val_harmless"] == 100
    assert report["harmbench_dropped_overlap_with_tests"] >= 1
    assert report["find_pool_dropped_overlap"] >= 1
    check_no_leaks(splits)


def test_build_splits_is_deterministic():
    a, _ = build_splits(SplitConfig(seed=0), loader=_fake_raw)
    b, _ = build_splits(SplitConfig(seed=0), loader=_fake_raw)
    assert a == b


def test_check_no_leaks_raises():
    with pytest.raises(AssertionError):
        check_no_leaks({"find_harmful": ["same prompt here"], "test_x": ["same prompt here"]})
