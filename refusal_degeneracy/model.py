"""Model loading, chat formatting, deterministic batched greedy generation."""
from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import Module

from .hooks import hooks


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(name: str, dtype: torch.dtype | None = None, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or default_device()
    dtype = dtype or default_dtype()
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def decoder_layers(model) -> Sequence[Module]:
    return model.model.layers


def chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode(tokenizer, prompts: Sequence[str], device: str):
    texts = [chat(tokenizer, p) for p in prompts]
    return tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


def batches(items: Sequence, size: int, progress: bool = False, desc: str | None = None) -> Iterator:
    indices = range(0, len(items), size)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices, desc=desc)
    for i in indices:
        yield items[i : i + size]


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    hook_spec: tuple | None = None,
    progress: bool = True,
) -> list[str]:
    out: list[str] = []
    ctx = hooks(*hook_spec) if hook_spec else nullcontext()
    with ctx:
        for batch in batches(prompts, batch_size, progress, "generate"):
            enc = encode(tokenizer, batch, model.device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            out += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return out
