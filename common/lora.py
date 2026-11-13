"""LoRA and OPLoRA adapters: ONE implementation for both, so the two methods differ ONLY in the projection.

Big picture
-----------
A frozen weight W0 (shape dout x din) maps an input x to W0 x.
LoRA adds a small trainable correction:   W = W0 + (alpha / r) * B A
    A : r x din     (maps the input down to r features)
    B : dout x r    (maps the r features back up to the output size)
OPLoRA uses the same A and B, but projects them before use:
    B~ = B - Uk (Uk^T B)      removes the part of B's columns lying in the top-k OUTPUT directions of W0
    A~ = A - (A Vk) Vk^T      removes the part of A's rows lying in the top-k INPUT directions of W0
    dW = (alpha / r) * B~ A~
Uk / Vk are the k leading left / right singular vectors of W0 (computed once, in FP32, frozen).
Consequence (OPLoRA Proposition 2): dW Vk = 0 and Uk^T dW = 0, so each top-k singular triple
(u_i, sigma_i, v_i) of W0 is still a singular triple of W0 + dW.

The dense projectors (I - Uk Uk^T) are never built; they are applied to the small factors.
"""
from __future__ import annotations

import hashlib
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.05,
                 A_init: torch.Tensor | None = None, Uk: torch.Tensor | None = None,
                 Vk: torch.Tensor | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dout, din = base.weight.shape
        dev = base.weight.device
        self.r, self.alpha, self.scale = r, float(alpha), float(alpha) / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # trainable factors are always FP32 ("FP32 adapters")
        if A_init is None:
            A_init = torch.empty(r, din)
            bound = 1.0 / math.sqrt(din)       # = kaiming_uniform(a=sqrt(5)), the LoRA/PEFT default
            A_init.uniform_(-bound, bound)
        self.lora_A = nn.Parameter(A_init.to(dev, torch.float32).clone())
        self.lora_B = nn.Parameter(torch.zeros(dout, r, device=dev, dtype=torch.float32))  # B = 0 -> dW = 0 at start
        # OPLoRA bases: frozen FP32 buffers, not trainable, not part of the adapter checkpoint
        if Uk is not None:
            self.register_buffer("Uk", Uk.to(dev, torch.float32).contiguous(), persistent=False)
            self.register_buffer("Vk", Vk.to(dev, torch.float32).contiguous(), persistent=False)
        else:
            self.Uk = None
            self.Vk = None

    @property
    def projected(self):
        return self.Uk is not None

    def effective_factors(self):
        """(A~, B~) in FP32. For plain LoRA these are just (A, B).
        Autocast is switched off so the projection itself is computed in FP32."""
        A, B = self.lora_A, self.lora_B
        if self.Uk is None:
            return A, B
        with torch.autocast(device_type=A.device.type, enabled=False):
            B_t = B - self.Uk @ (self.Uk.transpose(0, 1) @ B)     # dout x r
            A_t = A - (A @ self.Vk) @ self.Vk.transpose(0, 1)     # r x din
        return A_t, B_t

    def delta_weight(self):
        """Dense dW = (alpha/r) B~ A~ in FP32 (diagnostics / merging only)."""
        A_t, B_t = self.effective_factors()
        return self.scale * (B_t @ A_t)

    def forward(self, x):
        out = self.base(x)
        A_t, B_t = self.effective_factors()
        z = self.drop(x)
        if torch.is_autocast_enabled():
            upd = F.linear(F.linear(z, A_t), B_t)          # autocast runs these matmuls in FP16
        else:
            upd = F.linear(F.linear(z.to(A_t.dtype), A_t), B_t)
        return out + (self.scale * upd).to(out.dtype)


# ----------------------------------------------------------------------------- injection
def _iter_targets(model: nn.Module, targets):
    """Yield (parent, child_name, full_name, linear) in the model's fixed module order."""
    for full_name, mod in model.named_modules():
        for child_name, child in mod.named_children():
            if child_name in targets and isinstance(child, nn.Linear):
                yield mod, child_name, f"{full_name}.{child_name}" if full_name else child_name, child


BASIS_METHOD = "exact FP32 SVD, then a one-time FP64 QR re-orthonormalisation of Uk and Vk (column signs kept), stored in FP32"


@torch.no_grad()
def reorthonormalise(Q: torch.Tensor) -> torch.Tensor:
    """Same subspace, but with columns perpendicular to (almost) FP32 storage precision.
    An FP32 SVD of a 1,024 x 3,072 matrix returns singular vectors whose perpendicularity is off by ~1e-5;
    one QR in FP64 brings that to ~1e-6. Each column's sign is kept, so Q stays equal to the SVD vector
    up to ~1e-7 and (u_i, sigma_i, v_i) still pair up (needed for Proposition 2)."""
    Qd, R = torch.linalg.qr(Q.double())
    return (Qd * torch.sign(torch.diagonal(R))).to(Q.dtype).contiguous()


@torch.no_grad()
def topk_bases(W: torch.Tensor, k: int, device=None):
    """Leading k left/right singular vectors of W: exact FP32 SVD (not randomized), then re-orthonormalised once."""
    Wf = W.detach().to(device or W.device, torch.float32)
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    Uk = reorthonormalise(U[:, :k])
    Vk = reorthonormalise(Vh[:k].transpose(0, 1))
    return Uk, Vk, S


def inject_lora(model: nn.Module, targets, r: int, alpha: float, dropout: float, seed: int,
                protect_k: int = 0, svd_device=None, log=print):
    """Replace every target nn.Linear by a LoRALinear.

    A is drawn from ONE CPU generator seeded with `seed`, visiting modules in the model's fixed
    order. Two runs with the same rank and seed therefore get bit-identical initial A (this is how
    LoRA-16 and OPLoRA-16 share their "initial unprojected factors"). With protect_k > 0, each
    target also gets its top-k SVD bases (OPLoRA). Returns counts, timings and an A fingerprint.
    """
    g = torch.Generator().manual_seed(seed)
    h = hashlib.sha256()
    n_mod, n_params, basis_bytes, t_svd = 0, 0, 0, 0.0
    names = []
    for parent, child_name, full_name, lin in list(_iter_targets(model, targets)):
        dout, din = lin.weight.shape
        A0 = torch.empty(r, din)
        bound = 1.0 / math.sqrt(din)
        A0.uniform_(-bound, bound, generator=g)
        h.update(A0.numpy().tobytes())
        Uk = Vk = None
        if protect_k > 0:
            if protect_k >= min(dout, din):
                log(f"  WARNING: k={protect_k} >= min(dout, din)={min(dout, din)} for {full_name}: everything is protected")
            t0 = time.perf_counter()
            Uk, Vk, _ = topk_bases(lin.weight, protect_k, device=svd_device or lin.weight.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_svd += time.perf_counter() - t0
            basis_bytes += (Uk.numel() + Vk.numel()) * 4
        new = LoRALinear(lin, r, alpha, dropout, A_init=A0, Uk=Uk, Vk=Vk)
        setattr(parent, child_name, new)
        n_mod += 1
        n_params += r * (din + dout)
        names.append(full_name)
    info = {"n_modules": n_mod, "n_adapter_params": n_params, "init_A_sha256": h.hexdigest(),
            "basis_bytes": basis_bytes, "svd_seconds": t_svd, "module_names": names,
            "basis_method": BASIS_METHOD if protect_k > 0 else None}
    log(f"  injected {n_mod} adapters (r={r}, alpha={alpha}, k={protect_k}); "
        f"adapter params = {n_params:,}; initial-A fingerprint {info['init_A_sha256'][:16]}")
    return info


def lora_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def adapter_state(model, projected=True):
    """Export adapter factors in FP32. With projected=True, OPLoRA modules export (A~, B~),
    which is what inference and diagnostics must use."""
    out = {}
    for n, m in lora_modules(model).items():
        if projected:
            A_t, B_t = m.effective_factors()
        else:
            A_t, B_t = m.lora_A, m.lora_B
        out[f"{n}.A"] = A_t.detach().float().cpu().contiguous()
        out[f"{n}.B"] = B_t.detach().float().cpu().contiguous()
    return out


@torch.no_grad()
def merge_adapter_into(model: nn.Module, adapter: dict, scale: float, log=print):
    """Merge exported (already projected) factors into a fresh backbone:  W <- W0 + scale * B A.
    The sum is formed in FP32, then cast back to the backbone dtype. Returns the number of merged matrices."""
    mods = dict(model.named_modules())
    n = 0
    for key in adapter:
        if not key.endswith(".A"):
            continue
        name = key[:-2]
        lin = mods[name]
        if isinstance(lin, LoRALinear):
            lin = lin.base
        A = adapter[name + ".A"].to(lin.weight.device, torch.float32)
        B = adapter[name + ".B"].to(lin.weight.device, torch.float32)
        W = lin.weight.data.float() + scale * (B @ A)
        lin.weight.data.copy_(W.to(lin.weight.dtype))
        n += 1
    log(f"  merged {n} adapter matrices into the backbone (scale {scale})")
    return n


def attach_exported_adapter(model, adapter: dict, scale: float, dropout: float = 0.0):
    """Attach exported factors as (unmerged) plain LoRA modules, to compare merged vs unmerged logits.
    Projected factors need no further projection (projecting twice changes nothing)."""
    r = next(v for k, v in adapter.items() if k.endswith(".A")).shape[0]
    mods = dict(model.named_modules())
    for key in adapter:
        if not key.endswith(".A"):
            continue
        name = key[:-2]
        parent_name, child_name = name.rsplit(".", 1)
        parent = mods[parent_name]
        lin = getattr(parent, child_name)
        new = LoRALinear(lin, r, alpha=scale * r, dropout=dropout, A_init=adapter[key])
        with torch.no_grad():
            new.lora_B.copy_(adapter[name + ".B"].to(new.lora_B.device))
        setattr(parent, child_name, new)
    return model
