"""Device selection, precision contexts, synchronization and memory accounting."""
import contextlib
import gc

import torch

PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, precision):
    dtype = PRECISIONS[precision]
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


class PeakMemory:
    """Peak allocated bytes. Exact on CUDA (allocator high-water mark). MPS has
    no high-water-mark API, so there we sample current allocation at points
    the caller marks (e.g. end of forward, where activations peak)."""

    def __init__(self, device):
        self.device = device
        self.peak = 0

    def mark(self):
        if self.device.type == "mps":
            self.peak = max(self.peak, torch.mps.current_allocated_memory())

    def value(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated()
        return self.peak

    @property
    def exact(self):
        return self.device.type == "cuda"


def is_oom(err):
    msg = str(err).lower()
    return "out of memory" in msg or isinstance(err, torch.OutOfMemoryError)
