from refusal_degeneracy.config import ExperimentConfig, SplitConfig


def test_roundtrip(tmp_path):
    cfg = ExperimentConfig(model="org/M", split=SplitConfig(seed=7))
    path = cfg.to_json(tmp_path / "cfg.json")
    loaded = ExperimentConfig.from_json(path)
    assert loaded == cfg
    assert loaded.split.seed == 7
    assert isinstance(loaded.refusal_words, tuple)


def test_frozen():
    cfg = ExperimentConfig()
    try:
        cfg.model = "x"
    except Exception as e:
        assert "frozen" in type(e).__name__.lower() or "cannot assign" in str(e).lower()
    else:
        raise AssertionError("config should be frozen")
