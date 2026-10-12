import pytest
import torch


@pytest.fixture(scope="session")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    return model


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return {
        "input_ids": torch.randint(1, 64, (5, 6)),
        "attention_mask": torch.ones(5, 6, dtype=torch.long),
    }


@pytest.fixture
def dims():
    return {"d_model": 32, "d_mlp": 64, "n_layers": 4, "vocab": 64}
