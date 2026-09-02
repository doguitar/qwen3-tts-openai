"""Inference device selection. Pure helpers so they can be unit-tested without a GPU."""

from __future__ import annotations


def select_device(requested: str, cuda: bool, xpu: bool) -> str:
    """Pick a torch device string.

    Explicit TTS_DEVICE wins. Otherwise CUDA, then Intel XPU (Arc A380 and other
    Alchemist/Battlemage), then CPU.
    """
    requested = (requested or "").strip()
    if requested:
        return requested
    if cuda:
        return "cuda:0"
    if xpu:
        return "xpu"
    return "cpu"


def device_kind(device: str) -> str:
    return (device or "cpu").split(":", 1)[0].lower()


def inference_settings(device: str, dtype_override: str = "") -> tuple[str, str]:
    """Return (torch dtype name, attn_implementation).

    Arc A-series (Alchemist, including A380) has no native FP64 or BF16. Torch
    2.13+xpu float16 hits `torch._assert_async` in TensorCompareKernels during
    generate(); float32 + SDPA works. BF16 stays the CUDA default. CPU stays
    FP32 + eager attention.
    """
    kind = device_kind(device)
    override = (dtype_override or "").strip().lower()
    if override:
        dtype = override
    elif kind == "cuda":
        dtype = "bfloat16"
    elif kind == "xpu":
        dtype = "float32"
    else:
        dtype = "float32"
    attn = "sdpa" if kind in {"cuda", "xpu"} else "eager"
    return dtype, attn
