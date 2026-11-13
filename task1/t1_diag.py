"""Task 1 Step 6: weight-space and activation diagnostics on down_proj in layers 0, 13, 27 (FP32 throughout).

For each update dW (Full FT: W_trained - W0;  LoRA: (alpha/r) B A;  OPLoRA: (alpha/r) B~ A~):
    rho_k = ||Uk Uk^T dW||_F^2 / ||dW||_F^2     share of the update's energy inside the protected OUTPUT subspace
    eta   = ||dW||_F / ||W0||_F                  size of the update relative to the original matrix
    R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)     how strongly the update responds to a cached input z
z = the STARTING model's down_proj input at the last prompt token (benchmark prompt formats), cached once
with forward hooks and reused for every checkpoint. This isolates the local weight change; it ignores
changes elsewhere in the network.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.lora import topk_bases  # noqa: E402
from common.utils import Checks, read_jsonl, write_json  # noqa: E402
from task1.t1_data import MODEL_ID, build_prompt  # noqa: E402
from task1.t1_eval import find_run, load_hf  # noqa: E402

LAYERS = [0, 13, 27]
K = int(os.environ.get("PA3_T1_K", 128))   # env override: offline tests on a tiny model only
VARIANTS = [  # (label, method, source)
    ("full_mid", "full", "diag_down_proj_mid.pt"),
    ("full_final", "full", "diag_down_proj_final.pt"),
    ("l2sp_final", "l2sp", "diag_down_proj_final.pt"),
    ("lora4_final", "lora4", ("adapter_final_projected.pt", 8 / 4)),
    ("lora16_final", "lora16", ("adapter_final_projected.pt", 32 / 16)),
    ("oplora_final", "oplora", ("adapter_final_projected.pt", 32 / 16)),
]


def key(i):
    return f"model.layers.{i}.mlp.down_proj"


def probe_prompts(tok, probe):
    """The probe items in their benchmark prompt formats: GSM8K inside the non-thinking chat template,
    HellaSwag as the stock completion context (no chat template)."""
    return ([("math", build_prompt(tok, x["text"]), x["idx"]) for x in probe["gsm8k"]] +
            [("nonmath", x["text"], x["idx"]) for x in probe["hellaswag"]])


def check_probe_formats(tok, probe, baseline_eval_dir, checks):
    """Compare our probe prompts with the exact prompts lm-eval used for the same documents."""
    d = Path(baseline_eval_dir)
    g = {r["doc_id"]: r["prompt"] for r in read_jsonl(d / "gsm8k_samples.jsonl")}
    h = {r["doc_id"]: r["context"] for r in read_jsonl(d / "hellaswag_samples.jsonl")}
    ok_g = sum(g.get(x["idx"]) == build_prompt(tok, x["text"]) for x in probe["gsm8k"])
    ok_h = sum(h.get(x["idx"]) == x["text"] for x in probe["hellaswag"])
    checks.check(ok_g == 32, "probe GSM8K prompts are character-identical to lm-eval's prompts", "32 / 32", f"{ok_g} / 32")
    checks.check(ok_h == 32, "probe HellaSwag contexts are character-identical to lm-eval's contexts", "32 / 32", f"{ok_h} / 32")
    return ok_g == 32 and ok_h == 32


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", default="/kaggle/working/t1_diag")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    C = Checks()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = load_hf(MODEL_ID, dtype=torch.float32).to(dev).eval()      # BF16 checkpoint -> exact FP32
    W0 = {i: base.model.layers[i].mlp.down_proj.weight.detach().float().clone() for i in LAYERS}
    C.observe("down_proj shape (dout x din)", tuple(W0[0].shape))

    # ---- bases (same SVD routine as OPLoRA training)
    bases = {i: topk_bases(W0[i], K, device=dev) for i in LAYERS}
    op = find_run("oplora")
    if op is not None and (op / "diag_bases_down_proj.pt").exists():
        saved = torch.load(op / "diag_bases_down_proj.pt", map_location=dev)
        for i in LAYERS:
            Us, Uk = saved[key(i)]["Uk"].to(dev), bases[i][0]
            # compare subspaces through their projectors (insensitive to the sign of each singular vector)
            d = float((Us @ Us.T - Uk @ Uk.T).norm() / (Uk @ Uk.T).norm())
            C.check(d < 1e-3, f"layer {i}: OPLoRA's training bases span the same subspace as the diagnostic bases",
                    "relative projector difference < 1e-3", f"{d:.2e}")

    # ---- collect every dW
    deltas = {}
    for label, method, src in VARIANTS:
        run = find_run(method)
        if run is None:
            print(f"  (skipping {label}: run not attached)")
            continue
        if isinstance(src, str):
            p = run / src
            if not p.exists():
                print(f"  (skipping {label}: {p.name} missing)")
                continue
            sl = torch.load(p, map_location=dev)
            deltas[label] = {i: sl[key(i)].float().to(dev) - W0[i] for i in LAYERS}
        else:
            fname, scale = src
            ad = torch.load(run / fname, map_location=dev)
            deltas[label] = {i: scale * (ad[key(i) + ".B"].float() @ ad[key(i) + ".A"].float()) for i in LAYERS}

    rows = []
    for label, d in deltas.items():
        for i in LAYERS:
            dW, Uk, Vk = d[i], bases[i][0], bases[i][1]
            e = float(dW.pow(2).sum())
            rho = float((Uk @ (Uk.T @ dW)).pow(2).sum()) / e if e > 0 else float("nan")
            rho_right = float(((dW @ Vk) @ Vk.T).pow(2).sum()) / e if e > 0 else float("nan")
            sv2 = torch.linalg.svdvals(dW).pow(2)
            rows.append({"model": label, "layer": i, "rho_k": rho, "rho_k_input_side": rho_right,
                         "eta": float(dW.norm()) / float(W0[i].norm()), "dW_fro": float(dW.norm()),
                         "W0_fro": float(W0[i].norm()),
                         "dW_rank_99pct_energy": int((sv2.cumsum(0) / sv2.sum() < 0.99).sum().item() + 1)})
    if "oplora_final" in deltas:
        for r in rows:
            if r["model"] == "oplora_final":
                C.check(r["rho_k"] < 1e-6, f"OPLoRA layer {r['layer']}: rho_k ~ 0 (numerical check of the constraint)",
                        "< 1e-6", f"{r['rho_k']:.2e}")
        for i in LAYERS:
            Uk, Vk, S = bases[i]
            Wp = W0[i] + deltas["oplora_final"][i]
            err = float(((Wp @ Vk) - Uk * S[:K]).norm() / (Uk * S[:K]).norm())
            C.check(err < 1e-4, f"OPLoRA layer {i}: top-{K} singular triples preserved, (W0+dW) v_i = sigma_i u_i",
                    "relative error < 1e-4", f"{err:.2e}")

    # ---- activations: cache the STARTING model's down_proj inputs at the last prompt token
    probe = json.load(open(args.probe))
    prompts = probe_prompts(tok, probe)
    cache = {i: [] for i in LAYERS}
    cur, hooks = {}, []
    for i in LAYERS:
        def hook(mod, inp, i=i):
            cur[i] = inp[0][0, -1].detach().float().clone()       # batch of 1, no padding -> last position
        hooks.append(base.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook))
    for _dom, text, _idx in prompts:
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
        base(**enc)
        for i in LAYERS:
            cache[i].append(cur[i])
    for h in hooks:
        h.remove()
    Z = {i: torch.stack(cache[i]) for i in LAYERS}
    dom = np.array([p[0] for p in prompts])
    C.check(Z[0].shape == (64, W0[0].shape[1]), "cached 64 probe inputs per layer (32 math + 32 non-math)",
            (64, W0[0].shape[1]), tuple(Z[0].shape))
    torch.save({"Z": {i: Z[i].cpu() for i in LAYERS}, "domains": dom.tolist(), "ids": [p[2] for p in prompts]},
               out / "probe_activations.pt")

    arows, per_prompt = [], {}
    for label, d in deltas.items():
        for i in LAYERS:
            num = (Z[i] @ d[i].T).norm(dim=1)
            den = (Z[i] @ W0[i].T).norm(dim=1) + 1e-8
            R = (num / den).cpu().numpy()
            per_prompt[f"{label}|{i}"] = R.tolist()
            for dname in ["math", "nonmath"]:
                v = R[dom == dname]
                arows.append({"model": label, "layer": i, "domain": dname, "R_mean": float(v.mean()),
                              "R_median": float(np.median(v)), "R_std": float(v.std(ddof=1))})
            m, n = R[dom == "math"], R[dom == "nonmath"]
            arows.append({"model": label, "layer": i, "domain": "math/nonmath ratio of means", "R_mean": float(m.mean() / n.mean())})
    import pandas as pd
    wdf, adf = pd.DataFrame(rows), pd.DataFrame(arows)
    wdf.to_csv(out / "diag_weight_space.csv", index=False)
    adf.to_csv(out / "diag_activation.csv", index=False)
    write_json(out / "diag_activation_per_prompt.json", per_prompt)
    pd.set_option("display.width", 200)
    print(wdf.round(5).to_string(index=False))
    print(adf.pivot_table(index=["model", "layer"], columns="domain", values="R_mean").round(5).to_string())
    C.summary()
    C.save(out / "diag_checks.json")


if __name__ == "__main__":
    main()
