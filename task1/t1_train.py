"""Task 1 training: Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16 on 40,000 MetaMathQA examples.

Run one method per process (one GPU each):
    python -m task1.t1_train --method full   --data /kaggle/working/t1_data --out /kaggle/working/t1_runs
    python -m task1.t1_train --method oplora --data ... --out ... [--resume_from <recovery dir>] [--stop_at <unix time>]

Time limit: with --stop_at, a run that is still training at that time saves a recovery checkpoint, writes a
PAUSED file and exits with code 3, so the notebook can finish (and Kaggle can save the output) before the session
limit. Running again with that output attached resumes exactly where it stopped.
Crash safety: besides the four prescribed checkpoints, a run that has gone --safety_minutes (60) without a save
writes an extra recovery checkpoint (not logged as a scheduled one), so an unexpected crash loses at most ~1 h.
When resuming from an attached output, the checkpoint is first copied into the new output.

Disk: a full-model recovery checkpoint is ~7 GB. Only the newest one is kept while a run trains, and when a run
finishes all its recovery checkpoints are deleted (their sizes stay in checkpoints.jsonl), so all five runs fit
in Kaggle's 20 GB output. Use --keep_all_recovery to keep everything.

Objective (the class framework):   min  L_SFT(theta) + lambda * Omega(theta, theta0),   theta - theta0 in S
    Full FT : S = everything, lambda = 0
    L2-SP   : S = everything, lambda = 1e-3, Omega = 1/2 * sum_j (theta_j - theta0_j)^2 over ALL unique parameters
    LoRA    : S = {W0 + (alpha/r) B A} on the 7 target projections, lambda = 0
    OPLoRA  : S = {W0 + (alpha/r) B~ A~} with the top-128 singular directions protected, lambda = 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.lora import adapter_state, inject_lora, lora_modules  # noqa: E402
from common.utils import (SEED, append_jsonl, dir_size_bytes, env_info, gpu_peak, load_causal_lm,  # noqa: E402
                          read_jsonl, rng_state, seed_everything, set_rng_state, write_json)
from task1.t1_data import MODEL_ID, TrainData  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DIAG_LAYERS = [0, 13, 27]
METHODS = {
    "full":   dict(kind="full", lr=2e-5, lam=0.0),
    "l2sp":   dict(kind="full", lr=2e-5, lam=1e-3),
    "lora4":  dict(kind="lora", lr=2e-4, r=4, alpha=8, k=0),
    "lora16": dict(kind="lora", lr=2e-4, r=16, alpha=32, k=0),
    "oplora": dict(kind="lora", lr=2e-4, r=16, alpha=32, k=int(os.environ.get("PA3_T1_K", 128))),   # env override: tests only
}
SHARED = dict(n_updates=1250, micro=2, accum=16, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
              warmup=38, clip=1.0, max_len=2048, lora_dropout=0.05, seed=SEED,
              ckpt_updates=[312, 625, 937, 1250], loss_chunk=1024,
              schedule="transformers.get_cosine_schedule_with_warmup (linear 0 -> peak over 38 updates, cosine -> 0 at 1250)")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_model(kind):
    dtype = torch.float32 if kind == "full" else torch.float16   # FP32 master params vs frozen FP16 backbone
    if not torch.cuda.is_available():
        dtype = torch.float32                                     # CPU unit tests only
    return load_causal_lm(MODEL_ID, dtype)


def sft_loss_sum(model, input_ids, attn, labels, chunk):
    """SUM of next-token cross-entropy over supervised positions only (the caller divides by the
    effective-batch token count). The transformer body runs once; only hidden states whose NEXT token is
    supervised go through the output layer, in checkpointed chunks, so a 2048 x 151,936 logit matrix is
    never held in memory."""
    body = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    h = body[:, :-1]
    tgt = labels[:, 1:]
    sel = tgt != -100
    h, tgt = h[sel], tgt[sel]

    def ce(hc, tc):
        return F.cross_entropy(model.lm_head(hc).float(), tc, reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(ce, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
    return total, int(sel.sum())


def param_groups(named_trainable, wd):
    """Weight decay on matrices only; biases and parameters with fewer than two dimensions are excluded."""
    decay, no_decay = [], []
    for n, p in named_trainable:
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], len(decay), len(no_decay)


def make_optimizer(groups, lr):
    kw = dict(lr=lr, betas=SHARED["betas"], eps=SHARED["eps"])
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, fused=True, **kw), "fused"
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(groups, **kw), "default"


def cosine_with_warmup(opt, warmup, total):
    """Standard Hugging Face schedule: linear warm-up for 38 updates, then cosine decay to 0 (decision D2)."""
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(opt, warmup, total)


@torch.no_grad()
def l2sp_penalty_and_grad(params, refs, lam, add_grad=True, chunk=1 << 24):
    """Omega = 1/2 * sum_j (theta_j - theta0_j)^2  (a SUM over every unique parameter, never a mean).
    Its gradient lam * (theta - theta0) is added straight into .grad (after unscaling, before clipping).
    This is mathematically identical to backpropagating lam * Omega, without a 2.4 GB temporary.
    `refs` may live on the GPU or in pinned CPU memory (then each chunk is copied over)."""
    omega = torch.zeros((), dtype=torch.float64, device=params[0].device)
    for p, p0 in zip(params, refs):
        pv, rv = p.detach().view(-1), p0.view(-1)
        gv = p.grad.view(-1) if (add_grad and p.grad is not None) else None
        for i in range(0, pv.numel(), chunk):
            d = pv[i:i + chunk] - rv[i:i + chunk].to(device=pv.device, dtype=torch.float32, non_blocking=True)
            omega += d.double().pow(2).sum()
            if gv is not None and lam > 0:
                gv[i:i + chunk].add_(d, alpha=lam)
    return 0.5 * float(omega)


@torch.no_grad()
def displacement_cpu(model, ref_cpu):
    """||theta - theta0||_2 over all unique parameters, computed on the CPU."""
    s = 0.0
    for n, p in model.named_parameters():
        s += float((p.detach().float().cpu() - ref_cpu[n].float()).double().pow(2).sum())
    return math.sqrt(s)


def save_fp16_export(model, tok, out_dir):
    """FP16 copy of the full model for evaluation (the tied embedding / output layer is stored once)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd, seen = {}, {}
    for k, v in model.state_dict().items():
        ptr = v.data_ptr()
        if ptr in seen:
            sd[k] = sd[seen[ptr]]
        else:
            sd[k] = v.detach().to("cpu", torch.float16).contiguous()
            seen[ptr] = k
    model.save_pretrained(out_dir, state_dict=sd, safe_serialization=True)
    tok.save_pretrained(out_dir)
    return dir_size_bytes(out_dir)


def diag_slices(model):
    """FP32 copies of the diagnostic matrices (down_proj in layers 0, 13, 27)."""
    return {f"model.layers.{i}.mlp.down_proj": model.model.layers[i].mlp.down_proj.weight.detach().float().cpu().clone()
            for i in DIAG_LAYERS}


def choose_ref_storage(requested, dev):
    """Decision D3. 'auto' keeps theta0 in pinned CPU memory on GPUs under 20 GB (a 16 GB T4 is already
    ~13 GB full with Full FT), and on the GPU otherwise."""
    if requested != "auto":
        return requested
    if dev.type != "cuda":
        return "cpu_bf16"
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return "cpu_bf16" if total < 20 else "gpu_bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume_from", default=None, help="a recovery_u*/ folder (e.g. from an attached Kaggle output)")
    ap.add_argument("--ref_storage", default="auto", choices=["auto", "cpu_bf16", "gpu_bf16", "gpu_fp32"],
                    help="where Full FT keeps theta0 (decision D3)")
    ap.add_argument("--keep_all_recovery", action="store_true",
                    help="never delete recovery checkpoints (default: keep only the newest full-model one while training, "
                         "and delete all of a run's recovery checkpoints once it is finished)")
    ap.add_argument("--loss_chunk", type=int, default=SHARED["loss_chunk"],
                    help="positions per output-layer chunk; lower it (e.g. 512) only if Full FT runs out of GPU memory")
    ap.add_argument("--stop_at", type=float, default=None, help="unix time after which the run pauses itself (exit code 3)")
    ap.add_argument("--safety_minutes", type=float, default=60,
                    help="also save a (not logged) recovery checkpoint whenever this long has passed since the last save")
    ap.add_argument("--max_updates", type=int, default=None, help="debug only: stop early")
    ap.add_argument("--pause_after", type=int, default=None, help="debug only: pause after this many updates in this process")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    cfg = {**SHARED, **METHODS[args.method], "method": args.method, "model": MODEL_ID, "loss_chunk": args.loss_chunk}
    run = Path(args.out) / args.method
    run.mkdir(parents=True, exist_ok=True)
    total_u = args.max_updates or cfg["n_updates"]

    # ---------------------------------------------------------------- which recovery checkpoint to resume from
    rec = Path(args.resume_from) if args.resume_from else None
    if rec is None:
        cands = sorted([p for p in run.glob("recovery_u*") if (p / "state.pt").exists()],
                       key=lambda p: int(p.name.split("_u")[1]))
        rec = cands[-1] if cands else None
    if rec is not None and rec.parent.resolve() != run.resolve():
        # resuming from an attached (read-only) output: bring over logs, exports and diagnostic files, AND the
        # recovery checkpoint itself, so this session's output always holds the newest state even if it crashes
        for f in rec.parent.iterdir():
            if f.name.startswith("recovery_u") or f.name in ("PAUSED", "DONE") or (run / f.name).exists():
                continue
            (shutil.copytree if f.is_dir() else shutil.copy)(f, run / f.name)
        if not (run / rec.name / "state.pt").exists():
            shutil.copytree(rec, run / rec.name, dirs_exist_ok=True)
        log(f"copied {rec} into this session's output")
        rec = run / rec.name

    def carry_forward_and_pause(reason):
        """Exit without training (the newest state is already in this session's output)."""
        write_json(run / "PAUSED", {"update": int(rec.name.split("_u")[1]) if rec else 0, "reason": reason,
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"PAUSED before training ({reason}); run again with this output attached to continue")
        sys.exit(3)

    if args.stop_at and time.time() >= args.stop_at:
        carry_forward_and_pause("time limit reached before the run could start")

    seed_everything(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"=== {args.method} on {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}) ===")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    data = TrainData(Path(args.data) / "t1_train_tokens.npz", cfg["micro"], cfg["accum"])
    assert data.n_updates() == cfg["n_updates"], (data.n_updates(), cfg["n_updates"])

    overhead = {}
    t_load = time.perf_counter()
    model = load_model(cfg["kind"])
    model.config.use_cache = False
    model.to(dev)
    if cfg["kind"] == "lora":
        for p in model.parameters():          # freeze the WHOLE backbone (embeddings and norms included)
            p.requires_grad_(False)
        t0 = time.perf_counter()
        inj = inject_lora(model, TARGETS, cfg["r"], cfg["alpha"], cfg["lora_dropout"], seed=cfg["seed"],
                          protect_k=cfg["k"], svd_device=dev, log=log)
        overhead["adapter_injection_seconds"] = time.perf_counter() - t0
        if cfg["k"]:
            overhead["basis_svd_seconds"] = inj["svd_seconds"]
            overhead["basis_storage_bytes"] = inj["basis_bytes"]
        write_json(run / "injection.json", {k: v for k, v in inj.items() if k != "module_names"} |
                   {"n_names": len(inj["module_names"]), "first_names": inj["module_names"][:7]})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    overhead["model_load_seconds"] = time.perf_counter() - t_load

    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params_list = [p for _, p in named_trainable]
    n_trainable = sum(p.numel() for p in params_list)
    n_total = sum(p.numel() for p in model.parameters())
    groups, n_dec, n_nodec = param_groups(named_trainable, cfg["weight_decay"])
    opt, opt_impl = make_optimizer(groups, cfg["lr"])
    sched = cosine_with_warmup(opt, cfg["warmup"], cfg["n_updates"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    log(f"trainable params {n_trainable:,} / {n_total:,} (decay tensors {n_dec}, no-decay {n_nodec}); AdamW[{opt_impl}]")

    # ---------------------------------------------------------------- theta0 (Full FT variants)
    ref_cpu, refs = None, None
    if cfg["kind"] == "full":
        ref_cpu = {n: p.detach().to("cpu", torch.bfloat16).clone() for n, p in model.named_parameters()}
        exact = all(torch.equal(ref_cpu[n].float(), p.detach().float().cpu()) for n, p in model.named_parameters())
        log(f"[{'PASS' if exact else 'FAIL'}] BF16 copy of theta0 is bit-exact (the checkpoint is stored in BF16): {exact}")
        overhead["theta0_bf16_exact"] = exact
        if cfg["lam"] > 0:
            storage = choose_ref_storage(args.ref_storage, dev)
            if not exact and storage != "gpu_fp32":
                storage = "gpu_fp32"
            t0 = time.perf_counter()
            if storage == "cpu_bf16":
                refs = [ref_cpu[n].pin_memory() if dev.type == "cuda" else ref_cpu[n] for n, _ in named_trainable]
                ref_cpu = {n: r for (n, _), r in zip(named_trainable, refs)}      # one pinned copy serves both uses
            elif storage == "gpu_bf16":
                refs = [p.detach().to(torch.bfloat16).clone() for p in params_list]
            else:
                refs = [p.detach().clone() for p in params_list]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            overhead["reference_storage"] = storage
            overhead["reference_storage_bytes"] = sum(r.numel() * r.element_size() for r in refs)
            overhead["reference_setup_seconds"] = time.perf_counter() - t0
            overhead["reference_device"] = str(refs[0].device)
            if storage == "cpu_bf16" and dev.type == "cuda":      # cost of moving theta0 to the GPU once
                t0 = time.perf_counter()
                for r in refs:
                    r.to(dev, non_blocking=True)
                torch.cuda.synchronize()
                overhead["reference_transfer_seconds_per_update"] = time.perf_counter() - t0
            log(f"theta0 stored as {storage} on {overhead['reference_device']} "
                f"({overhead['reference_storage_bytes'] / 1e9:.2f} GB)")
    write_json(run / "config.json", {**cfg, "env": env_info(), "ref_storage_requested": args.ref_storage,
                                     "ref_storage_used": overhead.get("reference_storage")})

    # ---------------------------------------------------------------- state shared by the helpers below
    state = {"skipped": 0, "penalty_seconds": 0.0, "train_seconds": 0.0, "last_saved": None, "last_save_time": time.time()}

    def write_state(uu):
        """Atomic save: a crash mid-write never leaves a half-written state.pt."""
        ck = run / f"recovery_u{uu}"
        ck.mkdir(exist_ok=True)
        sd = ({k: v.detach().cpu() for k, v in model.state_dict().items()} if cfg["kind"] == "full"
              else {n: p.detach().cpu() for n, p in named_trainable})
        torch.save({"trainable": sd, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng": rng_state(), "update": uu, "examples_seen": uu * 32,
                    "data_position": {"next_update": uu, "next_example": uu * 32}, "skipped": state["skipped"],
                    "penalty_seconds": state["penalty_seconds"], "train_seconds": state["train_seconds"]},
                   ck / "state.pt.tmp")
        os.replace(ck / "state.pt.tmp", ck / "state.pt")
        del sd
        state["last_saved"] = uu
        state["last_save_time"] = time.time()
        if cfg["kind"] == "full" and not args.keep_all_recovery:     # each full-model recovery checkpoint is ~7 GB
            for old in run.glob("recovery_u*"):
                if old != ck:
                    shutil.rmtree(old, ignore_errors=True)
        return ck

    def after_checkpoint(uu, ck):
        """Everything a scheduled checkpoint records besides the resumable state itself."""
        info = {"update": uu, "examples_seen": uu * 32, "recovery_bytes": dir_size_bytes(ck / "state.pt"),
                "skipped_so_far": state["skipped"], "train_seconds_so_far": state["train_seconds"], **gpu_peak()}
        log(f"checkpoint u{uu}: recovery {info['recovery_bytes'] / 1e9:.2f} GB")
        if cfg["kind"] == "full":
            t0 = time.perf_counter()
            info["displacement_l2"] = displacement_cpu(model, ref_cpu)
            info["displacement_seconds"] = time.perf_counter() - t0
            log(f"  ||theta - theta0||_2 = {info['displacement_l2']:.4f}")
            tag = "mid" if (uu == 625 and args.method == "full") else "final" if uu == total_u else None
            if tag:                                    # only Full FT is evaluated at the midpoint
                info[f"export_{tag}_bytes"] = save_fp16_export(model, tok, run / f"export_{tag}")
                torch.save(diag_slices(model), run / f"diag_down_proj_{tag}.pt")
        else:
            t0 = time.perf_counter()
            torch.save(adapter_state(model, projected=True), ck / "adapter_projected.pt")
            if cfg["k"]:
                torch.save(adapter_state(model, projected=False), ck / "adapter_raw.pt")
            overhead["export_seconds"] = time.perf_counter() - t0
            info["adapter_bytes"] = (ck / "adapter_projected.pt").stat().st_size
            if uu == 625:
                shutil.copy(ck / "adapter_projected.pt", run / "adapter_mid_projected.pt")
            if uu == total_u:
                shutil.copy(ck / "adapter_projected.pt", run / "adapter_final_projected.pt")
        append_jsonl(run / "checkpoints.jsonl", info)

    def is_scheduled(uu):
        return uu in cfg["ckpt_updates"] or (args.max_updates is not None and uu == total_u)

    # ---------------------------------------------------------------- resume
    start_u = 0
    if rec is not None and (rec / "state.pt").exists():
        st = torch.load(rec / "state.pt", map_location="cpu", weights_only=False)
        res = model.load_state_dict(st["trainable"], strict=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start_u = st["update"]
        state.update(skipped=st["skipped"], penalty_seconds=st.get("penalty_seconds", 0.0),
                     train_seconds=st.get("train_seconds", 0.0), last_saved=start_u)
        for name in ["train_log.jsonl", "checkpoints.jsonl"]:   # drop rows written after the checkpoint (they are redone)
            p = run / name
            if p.exists():
                rows = [l for l in open(p) if l.strip() and json.loads(l)["update"] <= start_u]
                open(p, "w").writelines(rows)
        log(f"RESUMED from {rec} at update {start_u} (examples seen {start_u * 32}); "
            f"unexpected keys {len(res.unexpected_keys)}")
        del st
        # a crash between saving state.pt and finishing the checkpoint's other outputs: redo those outputs now
        logged = {json.loads(l)["update"] for l in open(run / "checkpoints.jsonl")} if (run / "checkpoints.jsonl").exists() else set()
        if is_scheduled(start_u) and start_u not in logged:
            after_checkpoint(start_u, rec)
    (run / "PAUSED").unlink(missing_ok=True)

    def pause(uu, reason):
        if uu > start_u and state["last_saved"] != uu:
            write_state(uu)
            log(f"pause checkpoint u{uu} saved")
        write_json(run / "PAUSED", {"update": uu, "reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"PAUSED at update {uu} ({reason}); run again with this output attached to continue")
        sys.exit(3)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    lam = cfg.get("lam", 0.0)
    t_run = time.perf_counter()
    for u in range(start_u, total_u):
        if args.stop_at and time.time() >= args.stop_at:
            pause(u, "time limit")
        if args.pause_after is not None and u - start_u >= args.pause_after:
            pause(u, "debug pause")
        t_u = time.perf_counter()
        n_tok = data.update_token_count(u)               # loss normaliser for the whole effective batch
        loss_sum_total = 0.0
        for j in range(cfg["accum"]):
            ids, attn, labels = (t.to(dev, non_blocking=True) for t in data.microbatch(u, j, tok.pad_token_id))
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                lsum, _ = sft_loss_sum(model, ids, attn, labels, cfg["loss_chunk"])
            scaler.scale(lsum / n_tok).backward()
            loss_sum_total += float(lsum)
        sft = loss_sum_total / n_tok
        scaler.unscale_(opt)                              # unscale BEFORE clipping
        omega = None
        if lam > 0:
            t_p = time.perf_counter()
            omega = l2sp_penalty_and_grad(params_list, refs, lam, add_grad=True)
            state["penalty_seconds"] += time.perf_counter() - t_p
        gnorm = float(torch.nn.utils.clip_grad_norm_(params_list, cfg["clip"]))
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skip = scaler.get_scale() < scale_before         # the scaler lowers its scale exactly when it skips a step
        state["skipped"] += int(skip)
        lr_used = sched.get_last_lr()[0]
        sched.step()
        opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t_u
        state["train_seconds"] += dt
        row = {"update": u + 1, "examples_seen": (u + 1) * 32, "lr": lr_used, "sft_loss": sft, "n_tokens": n_tok,
               "grad_norm_preclip": gnorm, "grad_scale": scale_before, "skipped": bool(skip), "seconds": dt}
        if omega is not None:
            row.update({"omega": omega, "lam_omega": lam * omega, "total_loss": sft + lam * omega,
                        "displacement_l2": math.sqrt(2 * omega)})
        append_jsonl(run / "train_log.jsonl", row)
        if (u + 1) % 25 == 0 or u == start_u:
            done_here = u + 1 - start_u
            eta = (total_u - u - 1) * (time.perf_counter() - t_run) / done_here / 3600
            extra = f" | omega {omega:.4g} lam*omega {lam * omega:.4g}" if omega is not None else ""
            log(f"u {u + 1:4d}/{total_u} | sft {sft:.4f} | lr {lr_used:.2e} | gnorm {gnorm:.3f} | "
                f"skipped {state['skipped']} | {dt:.1f}s/upd | peak {gpu_peak()['peak_allocated_gb']:.1f} GB | "
                f"ETA {eta:.2f} h{extra}")
        if is_scheduled(u + 1):
            ck = write_state(u + 1)
            after_checkpoint(u + 1, ck)
        elif u + 1 < total_u and time.time() - state["last_save_time"] >= args.safety_minutes * 60:
            write_state(u + 1)                              # safety checkpoint: resumable, not a scheduled one
            log(f"safety checkpoint u{u + 1} saved ({args.safety_minutes:g} min since the last save)")

    # ---------------------------------------------------------------- final artefacts
    train_seconds, skipped = state["train_seconds"], state["skipped"]
    if cfg["kind"] == "full" and not (run / "export_final").exists():
        # resumed from the final checkpoint after a crash during its export: the loaded model IS the final model
        save_fp16_export(model, tok, run / "export_final")
        torch.save(diag_slices(model), run / "diag_down_proj_final.pt")
    if cfg["kind"] == "lora" and not (run / "adapter_final_projected.pt").exists():
        torch.save(adapter_state(model, projected=True), run / "adapter_final_projected.pt")
    final = {"method": args.method, "trainable_params": n_trainable, "total_params": n_total,
             "train_seconds_measured": train_seconds, "wall_seconds_this_session": time.perf_counter() - t_run,
             "skipped_updates": skipped, "updates": total_u, **gpu_peak(), "overhead": overhead}
    if lam > 0:
        final["penalty_seconds"] = state["penalty_seconds"]
    ck_rows = read_jsonl(run / "checkpoints.jsonl") if (run / "checkpoints.jsonl").exists() else []
    final["recovery_checkpoint_bytes"] = max((r["recovery_bytes"] for r in ck_rows), default=None)
    if cfg["kind"] == "lora":
        final["adapter_bytes_fp32"] = (run / "adapter_final_projected.pt").stat().st_size
        if cfg["k"]:
            # numerical verification of the OPLoRA constraints on every adapted matrix
            worst_left = worst_right = 0.0
            for n, m in lora_modules(model).items():
                dW = m.delta_weight()
                nrm = float(dW.norm())
                if nrm < 1e-12:
                    continue
                worst_left = max(worst_left, float((m.Uk.T @ dW).norm()) / nrm)
                worst_right = max(worst_right, float((dW @ m.Vk).norm()) / nrm)
            final["oplora_worst_rel_UkT_dW"] = worst_left
            final["oplora_worst_rel_dW_Vk"] = worst_right
            log(f"OPLoRA constraint check: max ||Uk^T dW||/||dW|| = {worst_left:.2e}, "
                f"max ||dW Vk||/||dW|| = {worst_right:.2e} (expected ~1e-6 or smaller: FP32 round-off)")
            mods = lora_modules(model)
            torch.save({f"model.layers.{i}.mlp.down_proj": {"Uk": mods[f"model.layers.{i}.mlp.down_proj"].Uk.cpu(),
                                                              "Vk": mods[f"model.layers.{i}.mlp.down_proj"].Vk.cpu()}
                        for i in DIAG_LAYERS}, run / "diag_bases_down_proj.pt")
    else:
        final["export_final_bytes"] = dir_size_bytes(run / "export_final")
    write_json(run / "final_summary.json", final)
    if not args.keep_all_recovery:      # finished: recovery checkpoints are no longer needed (sizes are logged)
        for old in run.glob("recovery_u*"):
            shutil.rmtree(old, ignore_errors=True)
    (run / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"=== {args.method} finished: {train_seconds / 3600:.2f} h of training, peak "
        f"{final['peak_allocated_gb']:.2f} GB, skipped {skipped} ===")


if __name__ == "__main__":
    main()
