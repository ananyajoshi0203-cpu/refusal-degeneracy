import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_week1.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("run_week1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults(script):
    args = script.build_parser().parse_args([])
    assert args.config is None
    assert args.limit is None
    assert args.logits_only is False


def test_config_from_args_uses_defaults(script):
    args = script.build_parser().parse_args([])
    config = script.config_from_args(args)
    assert config.model == "Qwen/Qwen3-1.7B"
    assert config.split.seed == 0


def test_config_from_args_applies_overrides(script):
    args = script.build_parser().parse_args(
        ["--model", "sshleifer/tiny-gpt2", "--seed", "7", "--max-new-tokens", "32"]
    )
    config = script.config_from_args(args)
    assert config.model == "sshleifer/tiny-gpt2"
    assert config.split.seed == 7
    assert config.max_new_tokens == 32


def test_config_from_args_reads_file(script, tmp_path):
    from refusal_degeneracy.config import ExperimentConfig

    path = tmp_path / "c.json"
    replace(ExperimentConfig(), model="my/model").to_json(path)
    args = script.build_parser().parse_args(["--config", str(path)])
    assert script.config_from_args(args).model == "my/model"
