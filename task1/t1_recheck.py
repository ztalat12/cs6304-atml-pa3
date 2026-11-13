"""Task 1: redo of the diagnostics' Proposition-2 check for OPLoRA, with an exact (FP64) reference.

Why a redo. t1_diag.py checked (W0 + dW) v_i = sigma_i u_i for i <= k, using bases from an FP32 SVD on the GPU, and
reported FAIL in layers 0 and 13 (1.17e-4 and 1.08e-4 against a 1e-4 threshold). That error has two parts:

    (W0 + dW) Vk - Uk Sk  =  (W0 Vk - Uk Sk)   +   dW Vk
                              reference error       the update's effect

The first part does not involve training at all: it is how exactly the FP32 SVD satisfies its own definition.
Here the reference is an FP64 SVD of the BF16-exact W0 (accurate to ~1e-15), so what remains is the update's effect.

For every matrix OPLoRA adapted (196, so the three diagnostic down_proj matrices are included):
    right     = ||(W0 + dW) Vk - Uk Sk||_F / ||Sk||_F       Proposition 2, right side:  (W0 + dW) v_i = sigma_i u_i
    left      = ||Uk^T (W0 + dW) - Sk Vk^T||_F / ||Sk||_F   Proposition 2, left side:   u_i^T (W0 + dW) = sigma_i v_i^T
    reference = ||W0 Vk - Uk Sk||_F / ||Sk||_F              accuracy of the FP64 reference itself
PASS if right and left are below 1e-4, the original threshold (unchanged).
dW = (alpha / r) B~ A~ from the exported projected factors, multiplied in FP64.

With a GPU it also replays the original FP32 check exactly as t1_diag.py computed it and splits it into the two parts.
Nothing here changes the reported diagnostics (rho_k, eta, R(z)); the outputs go to their own folder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.lora import topk_bases  # noqa: E402
from common.utils import Checks, read_json  # noqa: E402
from task1.t1_data import MODEL_ID  # noqa: E402
from task1.t1_eval import find_run, load_hf  # noqa: E402

K = int(os.environ.get("PA3_T1_K", 128))   # env override: offline tests on a tiny model only
DIAG_LAYERS = [0, 13, 27]
THRESHOLD = 1e-4


def fp64_reference(W0: torch.Tensor, k: int):
    """Top-k singular vectors of W0 from an FP64 SVD: (Uk, all singular values S, Vk)."""
    U, S, Vh = torch.linalg.svd(W0.double(), full_matrices=False)
    return U[:, :k].contiguous(), S, Vh[:k].T.contiguous()


@torch.no_grad()
def prop2_errors(W0: torch.Tensor, dW: torch.Tensor, k: int) -> dict:
    """Proposition-2 errors for one matrix, everything in FP64 (see the module docstring)."""
    W, D = W0.double(), dW.double()
    Uk, S, Vk = fp64_reference(W, k)
    Sk = S[:k]
    nS = Sk.norm()
    UkSk = Uk * Sk
    Wp = W + D
    return {
        "right": float(((Wp @ Vk) - UkSk).norm() / nS),
        "left": float(((Uk.T @ Wp) - Sk[:, None] * Vk.T).norm() / nS),
        "update_right": float((D @ Vk).norm() / nS),
        "update_left": float((Uk.T @ D).norm() / nS),
        "reference": float(((W @ Vk) - UkSk).norm() / nS),
        "sigma_k": float(S[k - 1]),
        "boundary_gap_rel": float((S[k - 1] - S[k]) / S[k - 1]) if k < S.numel() else float("nan"),
        "eta": float(D.norm() / W.norm()),
    }


@torch.no_grad()
def original_fp32_check(W0: torch.Tensor, dW: torch.Tensor, k: int, device) -> dict:
    """The diagnostics' original check, computed as t1_diag.py did (FP32, topk_bases on `device`), split into the
    reference part and the update part. total <= reference_part + update_part (triangle inequality)."""
    W, D = W0.float().to(device), dW.float().to(device)
    Uk, Vk, S = topk_bases(W, k, device=device)
    UkSk = Uk * S[:k]
    n = UkSk.norm()
    return {"total": float(((W + D) @ Vk - UkSk).norm() / n),
            "reference_part": float((W @ Vk - UkSk).norm() / n),
            "update_part": float((D @ Vk).norm() / n)}


def original_values(path):
    """Layer -> value of the original FAIL/PASS lines in an attached diag_checks.json (if any)."""
    out = {}
    if path and Path(path).exists():
        for r in json.load(open(path)):
            m = re.search(r"OPLoRA layer (\d+): top-\d+ singular triples preserved", r["name"])
            if m:
                out[int(m.group(1))] = f"{r['actual']} ({r['status']})"
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/t1_recheck")
    ap.add_argument("--run", default=None, help="a finished t1_runs/oplora folder (default: the newest finished copy found)")
    ap.add_argument("--original_checks", default=None, help="diag_checks.json of the original diagnostics run (optional)")
    ap.add_argument("--replay_device", default=None,
                    help="device for replaying the original FP32 check (default: cuda when available, otherwise skipped)")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    C = Checks()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    run = Path(a.run) if a.run else find_run("oplora")
    ok = run is not None and (run / "adapter_final_projected.pt").exists()
    C.check(ok, "finished OPLoRA run with its exported projected factors", "t1_runs/oplora/adapter_final_projected.pt",
            str(run) if run else "not found: attach the t1_train output in which OPLoRA finished")
    if not ok:
        C.summary()
        C.save(out / "recheck_checks.json")
        return
    cfg = read_json(run / "config.json")
    scale = cfg["alpha"] / cfg["r"]
    inj = read_json(run / "injection.json") if (run / "injection.json").exists() else {}
    C.observe("OPLoRA run being checked", str(run), note=f"α / r = {scale:g}, k = {K}; bases: {inj.get('basis_method')}")

    ad = torch.load(run / "adapter_final_projected.pt", map_location="cpu")
    names = [n[:-2] for n in ad if n.endswith(".A")]
    base = load_hf(MODEL_ID, dtype=torch.float32)        # BF16 checkpoint -> exact FP32 (and then exact FP64)
    sd = base.state_dict()

    # ---- 1. the FP64 recheck on every adapted matrix
    rows, t0 = [], time.time()
    for j, n in enumerate(names):
        W0 = sd[n + ".weight"]
        dW = scale * (ad[n + ".B"].double() @ ad[n + ".A"].double())
        rows.append({"matrix": n, "shape": f"{W0.shape[0]} x {W0.shape[1]}", **prop2_errors(W0, dW, K)})
        if (j + 1) % 28 == 0 or j + 1 == len(names):
            print(f"  {j + 1:3d} / {len(names)} matrices  ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "oplora_prop2_recheck.csv", index=False)

    C.check(df.reference.max() < 1e-10, f"the FP64 reference is exact: W₀Vₖ = UₖΣₖ on all {len(df)} matrices",
            "< 1e-10", f"{df.reference.max():.1e}")
    for i in DIAG_LAYERS:
        r = df[df.matrix == f"model.layers.{i}.mlp.down_proj"]
        if r.empty:
            continue
        r = r.iloc[0]
        C.check(max(r.right, r.left) < THRESHOLD,
                f"redo of the diagnostics check, layer {i} down_proj: top-{K} singular triples preserved (FP64 reference)",
                f"(W₀ + ΔW) vᵢ = σᵢ uᵢ and uᵢᵀ (W₀ + ΔW) = σᵢ vᵢᵀ, relative error < {THRESHOLD:g}",
                f"{r.right:.2e} (right side), {r.left:.2e} (left side)")
    wr, wl = df.loc[df.right.idxmax()], df.loc[df.left.idxmax()]
    C.check(max(df.right.max(), df.left.max()) < THRESHOLD,
            f"all {len(df)} OPLoRA matrices: top-{K} singular triples preserved (FP64 reference)", f"< {THRESHOLD:g}",
            f"worst right side {wr.right:.2e} ({wr.matrix}), worst left side {wl.left:.2e} ({wl.matrix})")
    C.observe("median error over all matrices (right side / left side)", f"{df.right.median():.1e} / {df.left.median():.1e}")
    g = df.loc[df.boundary_gap_rel.idxmin()]
    C.observe("smallest relative gap between σₖ and σₖ₊₁", f"{g.boundary_gap_rel:.1e} ({g.matrix})",
              note="a tiny gap makes the protected subspace sensitive to rounding in the FP32 training bases")

    # ---- 2. replay of the original FP32 check (GPU, as in t1_diag.py), split into its two parts
    replay = a.replay_device or ("cuda" if torch.cuda.is_available() else None)
    if replay:
        dev = torch.device(replay)
        orig = original_values(a.original_checks)
        rep = []
        for i in DIAG_LAYERS:
            n = f"model.layers.{i}.mlp.down_proj"
            if n + ".A" not in ad:
                continue
            dW32 = scale * (ad[n + ".B"].float().to(dev) @ ad[n + ".A"].float().to(dev))
            o = original_fp32_check(sd[n + ".weight"], dW32, K, dev)
            rep.append({"layer": i, **o, "original_run": orig.get(i)})
            C.observe(f"layer {i}: original FP32 check replayed on {dev.type}, and its two parts",
                      f"total {o['total']:.2e}; reference alone {o['reference_part']:.2e}; update alone {o['update_part']:.2e}",
                      note=f"original run: {orig.get(i, 'not attached')}")
        pd.DataFrame(rep).to_csv(out / "original_check_replay.csv", index=False)
    else:
        print("(no GPU: the replay of the original FP32 GPU check is skipped)")

    pd.set_option("display.width", 200)
    cols = ["matrix", "shape", "right", "left", "update_right", "update_left", "reference", "boundary_gap_rel", "eta"]
    print("\nworst 8 matrices by right-side error:")
    print(df.sort_values("right", ascending=False)[cols].head(8).to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    C.summary()
    C.save(out / "recheck_checks.json")


if __name__ == "__main__":
    main()
