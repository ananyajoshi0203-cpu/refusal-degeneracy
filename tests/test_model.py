from refusal_degeneracy.model import batches, default_device, default_dtype, set_seed


def test_batches_cover_all():
    items = list(range(10))
    out = [b for b in batches(items, 3)]
    assert sum(len(b) for b in out) == 10
    assert out[0] == [0, 1, 2]


def test_set_seed_reproducible():
    import torch

    set_seed(0)
    a = torch.rand(3)
    set_seed(0)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_device_and_dtype_are_valid():
    assert default_device() in {"cpu", "cuda"}
    assert default_dtype() is not None
