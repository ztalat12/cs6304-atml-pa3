"""Shared helpers used by every task.

The `Checks` printer is the "intended vs actual" output style used in every notebook:
each check prints PASS / FAIL / OBSERVE, then what we expected and what we got.
    PASS / FAIL  -> there is a right answer (a shape, a count, a constraint)
    OBSERVE      -> a measurement with no right answer (the report interprets it)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SEED = 6304


# ----------------------------------------------------------------------------- checks
class Checks:
    """Collects PASS / FAIL / OBSERVE lines so a notebook can print a summary at the end."""

    def __init__(self, stop_on_fail: bool = False):
        self.rows = []
        self.stop_on_fail = stop_on_fail

    def _emit(self, status, name, expected, actual):
        print(f"[{status:<7}] {name}")
        if expected is not None:
            print(f"          expected: {expected}")
        if actual is not None:
            print(f"          actual  : {actual}")
        self.rows.append({"status": status, "name": name,
                          "expected": None if expected is None else str(expected),
                          "actual": None if actual is None else str(actual)})

    def check(self, cond: bool, name: str, expected=None, actual=None):
        status = "PASS" if cond else "FAIL"
        self._emit(status, name, expected, actual)
        if not cond and self.stop_on_fail:
            raise AssertionError(f"check failed: {name}")
        return cond

    def observe(self, name: str, actual=None, note=None):
        """A measurement with no right answer (the report interprets it)."""
        self._emit("OBSERVE", name, note, actual)

    def summary(self):
        n_fail = sum(r["status"] == "FAIL" for r in self.rows)
        n_pass = sum(r["status"] == "PASS" for r in self.rows)
        n_obs = sum(r["status"] == "OBSERVE" for r in self.rows)
        print(f"\n==== {n_pass} PASS | {n_fail} FAIL | {n_obs} OBSERVE ====")
        for r in self.rows:
            if r["status"] == "FAIL":
                print("  FAIL:", r["name"])
        return n_fail

    def save(self, path):
        write_json(path, self.rows)


# ----------------------------------------------------------------------------- seeding
def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rng_state():
    """Every random-number-generator state needed to resume exactly."""
    import torch
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    import torch
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"])
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["torch_cuda"])


# ----------------------------------------------------------------------------- io
def write_json(path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_size_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human_bytes(n):
    n = float(n or 0)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


# ----------------------------------------------------------------------------- timing / memory
@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def gpu_peak(reset=False):
    import torch
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    out = {"peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
           "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return out


def env_info():
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["gpu_total_gb"] = [torch.cuda.get_device_properties(i).total_memory / 1e9
                                    for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    import importlib.metadata as md
    for pkg in ["transformers", "datasets", "lm-eval", "tokenizers", "numpy", "pandas", "safetensors"]:
        try:
            info[pkg] = md.version(pkg)
        except Exception:
            info[pkg] = None
    return info


# ----------------------------------------------------------------------------- model loading
def load_causal_lm(path_or_id, dtype, attn_implementation="sdpa", **extra):
    """Load a causal LM in exactly `dtype`. transformers renamed `torch_dtype` to `dtype` in 4.56; on older
    versions an unknown `dtype` keyword is silently ignored (the model would load in FP32), so pick the keyword
    by version and then verify the dtype actually loaded. `extra` is passed on (e.g. revision=...)."""
    import importlib.metadata as md

    from transformers import AutoModelForCausalLM
    v = tuple(int(x) for x in md.version("transformers").split(".")[:2])
    kw = {"dtype": dtype} if v >= (4, 56) else {"torch_dtype": dtype}
    m = AutoModelForCausalLM.from_pretrained(path_or_id, attn_implementation=attn_implementation, **kw, **extra)
    got = next(m.parameters()).dtype
    assert got == dtype, f"model loaded as {got}, expected {dtype}"
    return m
