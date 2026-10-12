from refusal_degeneracy.runs import load_run, results_root, run_path, save_run


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
