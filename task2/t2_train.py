"""Task 2 training: three-stage curriculum for the SigLIP -> connector -> Qwen3 VLM.

    Stage 1  caption alignment           2,250 updates  connector only (peak LR 1e-3)
    Stage 2  visual instruction tuning   1,500 updates  connector (1e-4) + LoRA r16 on q/k/v/o (2e-5)
             draws: presence 50% | counting 25% | caption replay 25%
    Stage 3  natural VQA specialisation    750 updates  connector (5e-5) + LoRA (1e-5)
             draws: VQAv2 70% | presence 10% | counting 10% | caption 10%
Every draw contributes two examples (a complete question pair, or two captions for replay); 8 draws per update.
Effective batch 16 = 2 microbatches x 8. Loss = mean over the 16 examples of each example's mean token NLL.
Each stage starts a fresh AdamW (its parameter set and learning rates change) with 5% warm-up, then cosine decay to
10% of the stage's peak learning rate.

    python -m task2.t2_train --sel <t2_selection.json> --feats <feature dir> --out /kaggle/working/t2/run196 [--tokens 49]

Checkpoints: resumable state every 250 updates (and at stage ends), connector/adapter weights after each stage.
Time limit: with --stop_at the run saves a checkpoint, writes PAUSED and exits with code 3; running again resumes.
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

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.lora import inject_lora, lora_modules  # noqa: E402
from common.utils import (SEED, append_jsonl, dir_size_bytes, env_info, gpu_peak, read_json,  # noqa: E402
                          rng_state, seed_everything, set_rng_state, write_json)
from task2.t2_features import FeatureStore  # noqa: E402
from task2.t2_model import LM_ID, VLM, Connector, SeqBuilder, calibrate_scale, collate, load_lm  # noqa: E402

STAGES = [
    dict(name="caption_alignment", updates=2250, lr_conn=1e-3, lr_lora=None, mix=None),
    dict(name="visual_instruction", updates=1500, lr_conn=1e-4, lr_lora=2e-5,
         mix=[("presence", 0.50), ("counting", 0.25), ("caption", 0.25)]),
    dict(name="natural_vqa", updates=750, lr_conn=5e-5, lr_lora=1e-5,
         mix=[("vqa", 0.70), ("presence", 0.10), ("counting", 0.10), ("caption", 0.10)]),
]
SHARED = dict(micro=8, accum=2, weight_decay=0.01, clip=1.0, warmup_frac=0.05, final_lr_frac=0.10,
              lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
              ckpt_every=250, dev_every=500, seed=SEED, grounding_sampling="category_balanced",
              optimizer="fresh AdamW (betas 0.9/0.999, eps 1e-8) at the start of each stage")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stage_bounds():
    out, s = [], 0
    for st in STAGES:
        out.append((s, s + st["updates"]))
        s += st["updates"]
    return out


def dev_points():
    ends = [b for _, b in stage_bounds()]
    step = SHARED["dev_every"]
    return sorted(set([0] + list(range(step, ends[-1] + 1, step)) + ends))


def lr_lambda_factory(total, warmup_frac, final_frac):
    """Multiplier of the stage's peak LR after `step` updates of this stage: linear warm-up over ceil(5%) updates,
    then cosine decay from 1 to final_frac (0.1) at the stage's last update."""
    W = math.ceil(warmup_frac * total)

    def f(step):
        if step < W:
            return (step + 1) / W
        prog = (step - W) / max(1, total - 1 - W)
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f, W


class Data:
    """All training examples as (kind, image_key, question, target); tokenisation is cached."""

    def __init__(self, sel, builder, n_vis, grounding_sampling):
        self.b, self.n_vis = builder, n_vis
        self.captions = [("caption", e["image"], None, e["caption"]) for e in sel["caption_train"]]
        self.pools = {}
        for kind in ["presence", "counting"]:
            pairs = [[(q["type"], q["image"], q["question"], q["answer"], q["cat"]) for q in pr]
                     for pr in sel["grounding_train_pool"][kind]]
            by_cat = {}
            for p in pairs:
                by_cat.setdefault(p[0][4], []).append(p)
            self.pools[kind] = {"pairs": pairs, "by_cat": by_cat, "cats": sorted(by_cat)}
        self.vqa = [[("vqa", q["image"], q["question"], q["target"], q["type"]) for q in pr] for pr in sel["vqa_train"]]
        self.sampling = grounding_sampling
        self._cache = {}

    def tok(self, ex, n_vis=None, with_target=True):
        n_vis = self.n_vis if n_vis is None else n_vis
        kind = "caption" if ex[0] == "caption" else "qa"
        key = (kind, ex[2], ex[3] if with_target else None, n_vis)
        if key not in self._cache:
            self._cache[key] = self.b.build(kind, n_vis, question=ex[2], target=ex[3] if with_target else None)
        return self._cache[key]

    def draw(self, rng, mix):
        names = [m for m, _ in mix]
        k = names[rng.choice(len(names), p=np.array([p for _, p in mix]))]
        if k == "caption":
            i, j = rng.choice(len(self.captions), 2, replace=False)
            return k, [self.captions[i], self.captions[j]]
        if k == "vqa":
            p = self.vqa[rng.integers(len(self.vqa))]
            return k, [x[:4] for x in p]
        pool = self.pools[k]
        if self.sampling == "category_balanced":       # D-T2-3: a category uniformly, then one of its pairs
            cand = pool["by_cat"][pool["cats"][rng.integers(len(pool["cats"]))]]
            p = cand[rng.integers(len(cand))]
        else:
            p = pool["pairs"][rng.integers(len(pool["pairs"]))]
        return k, [x[:4] for x in p]


def make_batch(data, feats, examples, dev, n_vis=None, with_target=True, left_pad=False):
    items = [data.tok(e, n_vis=n_vis, with_target=with_target) for e in examples]
    batch = collate(items, data.b.pad, left_pad=left_pad)
    nv = data.n_vis if n_vis is None else n_vis
    f = feats.get([e[1] for e in examples]) if nv > 0 else None
    return {k: v.to(dev) for k, v in batch.items()}, f


def dev_questions(sel, suite):
    """(question examples with own image, same with partner image) for a development suite."""
    qs, partner = [], []
    for pr in sel[suite]:
        for j, q in enumerate(pr):
            ref = q["target"] if "target" in q else q["answer"]
            qs.append(("qa", q["image"], q["question"], ref))
            partner.append(("qa", pr[1 - j]["image"], q["question"], ref))
    return qs, partner


@torch.no_grad()
def dev_eval(vlm, data, feats, sel, dev, bs=8):
    """Caption reference loss + reference-answer NLL (termination excluded) on both development suites under
    correct / complementary / no-image inputs. Teacher-forced; nothing is generated."""
    vlm.eval()
    out = {}
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        caps = [("caption", e["image"], None, e["caption"]) for e in sel["caption_dev"]]
        v = []
        for i in range(0, len(caps), bs):
            b, f = make_batch(data, feats, caps[i:i + bs], dev)
            v.append(vlm.example_nll(b, f))
        out["caption_dev_loss"] = float(torch.cat(v).mean())
        for suite in ["grounding_dev", "vqa_dev"]:
            qs, partner = dev_questions(sel, suite)
            res = {}
            for cond, exs, nv in [("correct", qs, None), ("complementary", partner, None), ("no_image", qs, 0)]:
                v = []
                for i in range(0, len(exs), bs):
                    b, f = make_batch(data, feats, exs[i:i + bs], dev, n_vis=nv)
                    v.append(vlm.example_nll(b, f, exclude_last=True))
                res[cond] = torch.cat(v).cpu().numpy()
            for cond in res:
                out[f"{suite}_nll_{cond}"] = float(res[cond].mean())
            out[f"{suite}_swap_penalty"] = float((res["complementary"] - res["correct"]).mean())
            out[f"{suite}_no_image_penalty"] = float((res["no_image"] - res["correct"]).mean())
    vlm.train()
    return out


def build_model(dev, sel, data, feats, log=log):
    """Frozen Qwen + fresh connector (seeded), output scale calibrated on TRAINING data only."""
    lm = load_lm(dev)
    lm.config.use_cache = False
    vlm = VLM(lm, Connector(d_out=lm.config.hidden_size).to(dev))
    rng = np.random.default_rng([SEED, 99])
    texts = [e["caption"] for e in sel["caption_train"][:1500]] + [q["question"] for pr in sel["vqa_train"][:250] for q in pr]
    img_keys = sorted({e["image"] for e in sel["caption_train"]})
    pick = [img_keys[i] for i in rng.choice(len(img_keys), min(512, len(img_keys)), replace=False)]
    cal = calibrate_scale(vlm, data.b, texts, feats.get(pick).float(), dev, log=log)
    return vlm, cal


def optimizer_for(vlm, st):
    groups = []
    for name, params, lr in [("connector", list(vlm.connector.named_parameters()), st["lr_conn"]),
                             ("lora", [(n, p) for n, p in vlm.lm.named_parameters() if p.requires_grad], st["lr_lora"])]:
        if not params or lr is None:
            continue
        dec = [p for n, p in params if p.ndim >= 2]
        nod = [p for n, p in params if p.ndim < 2]
        groups.append({"params": dec, "lr": lr, "weight_decay": SHARED["weight_decay"], "name": name + "_decay"})
        if nod:
            groups.append({"params": nod, "lr": lr, "weight_decay": 0.0, "name": name + "_nodecay"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def trainable(vlm):
    return [p for p in vlm.parameters() if p.requires_grad]


def trainable_state(vlm):
    sd = {n: p.detach().cpu() for n, p in vlm.named_parameters() if p.requires_grad}
    sd["connector.scale"] = vlm.connector.scale.detach().cpu()
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=196, choices=[196, 49])
    ap.add_argument("--resume_from", default=None, help="a ckpt_u*/ folder (e.g. from an attached Kaggle output)")
    ap.add_argument("--stop_at", type=float, default=None, help="unix time after which the run pauses itself (exit code 3)")
    ap.add_argument("--safety_minutes", type=float, default=60, help="extra checkpoint when this long passed since the last save")
    ap.add_argument("--debug_stage_updates", default=None, help="debug only, e.g. 3,2,2")
    ap.add_argument("--debug_ckpt_every", type=int, default=None, help="debug only")
    ap.add_argument("--debug_dev_every", type=int, default=None, help="debug only")
    ap.add_argument("--pause_after", type=int, default=None, help="debug only: pause after this many updates in this process")
    args = ap.parse_args()
    if args.debug_stage_updates:
        for st, n in zip(STAGES, args.debug_stage_updates.split(",")):
            st["updates"] = int(n)
    if args.debug_ckpt_every:
        SHARED["ckpt_every"] = args.debug_ckpt_every
    if args.debug_dev_every:
        SHARED["dev_every"] = args.debug_dev_every
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bounds = stage_bounds()
    total = bounds[-1][1]
    devpts = set(dev_points())

    # ---------------------------------------------------------------- which checkpoint to resume from
    rec = Path(args.resume_from) if args.resume_from else None
    if rec is None:
        c = sorted([p for p in out.glob("ckpt_u*") if (p / "state.pt").exists()], key=lambda p: int(p.name[6:]))
        rec = c[-1] if c else None
    if rec is not None and rec.parent.resolve() != out.resolve():
        # attached (read-only) output: copy logs, stage weights etc. AND the checkpoint itself into this output
        for f in rec.parent.iterdir():
            if f.name.startswith("ckpt_u") or f.name in ("PAUSED", "DONE") or (out / f.name).exists():
                continue
            (shutil.copytree if f.is_dir() else shutil.copy)(f, out / f.name)
        if not (out / rec.name / "state.pt").exists():
            shutil.copytree(rec, out / rec.name, dirs_exist_ok=True)
        log(f"copied {rec} into this session's output")
        rec = out / rec.name
    if args.stop_at and time.time() >= args.stop_at:
        write_json(out / "PAUSED", {"global_update": int(rec.name[6:]) if rec else 0, "reason": "time limit before start"})
        log("PAUSED before training (time limit); run again with this output attached")
        sys.exit(3)

    seed_everything(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel = json.load(open(args.sel))
    feats = FeatureStore(args.feats, pool49=args.tokens == 49)
    tok = AutoTokenizer.from_pretrained(LM_ID)
    builder = SeqBuilder(tok)
    data = Data(sel, builder, args.tokens, SHARED["grounding_sampling"])
    write_json(out / "config.json", {"stages": STAGES, **SHARED, "tokens": args.tokens, "lm": LM_ID, "env": env_info(),
                                     "rendered_prompt_template": builder.rendered_example,
                                     "placeholder_token_id": builder.placeholder})
    vlm, cal = build_model(dev, sel, data, feats)
    write_json(out / "calibration.json", cal)
    vlm.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.train()
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    resume = None
    g = 0
    if rec is not None and (rec / "state.pt").exists():
        resume = torch.load(rec / "state.pt", map_location="cpu", weights_only=False)
        g = resume["global_update"]
        log(f"RESUMING from {rec} at global update {g}")
        for name in ["train_log.jsonl", "dev_log.jsonl", "checkpoints.jsonl"]:   # drop rows written after the checkpoint
            p = out / name
            if p.exists():
                keep = [l for l in open(p) if l.strip() and json.loads(l)["global_update"] <= g]
                open(p, "w").writelines(keep)
    (out / "PAUSED").unlink(missing_ok=True)
    g_start = g
    stage_stats = read_json(out / "stage_stats.json") if (out / "stage_stats.json").exists() else {}
    t_session = time.perf_counter()
    last_save = {"g": g, "t": time.time()}

    def save_ckpt(si, s0, opt, sched, rng, mix):
        ck = out / f"ckpt_u{g}"
        ck.mkdir(exist_ok=True)
        torch.save({"trainable": trainable_state(vlm), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "sampler_state": rng.bit_generator.state, "stage": si,
                    "stage_progress": g - s0, "global_update": g, "mix": mix, "rng": rng_state()}, ck / "state.pt.tmp")
        os.replace(ck / "state.pt.tmp", ck / "state.pt")
        last_save.update(g=g, t=time.time())
        write_json(out / "stage_stats.json", stage_stats)
        return ck

    def ensure_lora():
        if not lora_modules(vlm.lm):
            info = inject_lora(vlm.lm, SHARED["lora_targets"], SHARED["lora_r"], SHARED["lora_alpha"],
                               SHARED["lora_dropout"], seed=SEED, log=log)
            write_json(out / "lora_injection.json", {k: v for k, v in info.items() if k != "module_names"})

    for si, st in enumerate(STAGES):
        s0, s1 = bounds[si]
        if g >= s1 and not (resume is not None and resume["stage"] == si):
            if si >= 1:
                ensure_lora()            # stage already finished earlier: later stages still need the adapters
            continue
        if si >= 1:
            ensure_lora()
        if resume is not None:
            res = vlm.load_state_dict(resume["trainable"], strict=False)
            assert not res.unexpected_keys, res.unexpected_keys
        opt = optimizer_for(vlm, st)
        lam, W = lr_lambda_factory(st["updates"], SHARED["warmup_frac"], SHARED["final_lr_frac"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
        rng = np.random.default_rng([SEED, si + 1])
        order = None
        if si == 0:          # two passes over the captions, reshuffled each pass
            n = len(data.captions)
            order = np.concatenate([rng.permutation(n), rng.permutation(n)])
        mix = {"draws": {}, "examples": {}}
        if resume is not None and resume["stage"] == si:
            opt.load_state_dict(resume["optimizer"])
            sched.load_state_dict(resume["scheduler"])
            scaler.load_state_dict(resume["scaler"])
            rng.bit_generator.state = resume["sampler_state"]
            mix = resume["mix"]
            set_rng_state(resume["rng"])
            resume = None
        n_tr = sum(p.numel() for p in trainable(vlm))
        log(f"=== stage {si + 1} ({st['name']}): updates {s0}->{s1}, trainable params {n_tr:,}, warm-up {W} ===")
        ss = stage_stats.setdefault(str(si + 1), {"name": st["name"], "seconds": 0.0, "dev_eval_seconds": 0.0, "peak_allocated_gb": 0.0})
        ss.update({"trainable_params": n_tr, "connector_params": sum(p.numel() for p in vlm.connector.parameters()),
                   "lora_params": sum(p.numel() for n, p in vlm.lm.named_parameters() if p.requires_grad), "warmup_updates": W})
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if g == 0 and 0 in devpts and not (out / "dev_log.jsonl").exists():
            t0 = time.perf_counter()
            d = dev_eval(vlm, data, feats, sel, dev)
            ss["dev_eval_seconds"] += time.perf_counter() - t0
            append_jsonl(out / "dev_log.jsonl", {"global_update": 0, "stage": 0, **d})
            log("dev @0: " + ", ".join(f"{k}={v:.4f}" for k, v in d.items()))
        while g < s1:
            # ---- time limit / debug pause (checked before starting an update)
            reason = ("time limit" if args.stop_at and time.time() >= args.stop_at else
                      "debug pause" if args.pause_after is not None and g - g_start >= args.pause_after else None)
            if reason:
                if last_save["g"] != g:
                    save_ckpt(si, s0, opt, sched, rng, mix)
                write_json(out / "PAUSED", {"global_update": g, "reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                log(f"PAUSED at global update {g} ({reason}); run again with this output attached to continue")
                sys.exit(3)
            t0 = time.perf_counter()
            k = g - s0
            if si == 0:
                exs = [data.captions[i] for i in order[k * 16:(k + 1) * 16]]
                kinds = ["caption"] * 16
            else:
                exs, kinds = [], []
                for _ in range(8):
                    kind, two = data.draw(rng, st["mix"])
                    mix["draws"][kind] = mix["draws"].get(kind, 0) + 1
                    exs += two
                    kinds += [kind, kind]
            for kd in kinds:
                mix["examples"][kd] = mix["examples"].get(kd, 0) + 1
            loss_total = 0.0
            for mb in range(SHARED["accum"]):
                b, f = make_batch(data, feats, exs[mb * 8:(mb + 1) * 8], dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    nll = vlm.example_nll(b, f)
                scaler.scale(nll.sum() / 16.0).backward()
                loss_total += float(nll.sum())
            scaler.unscale_(opt)                        # unscale BEFORE clipping
            gn = float(torch.nn.utils.clip_grad_norm_(trainable(vlm), SHARED["clip"]))
            sc = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            lrs = {gr["name"]: gr["lr"] for gr in opt.param_groups}
            sched.step()
            opt.zero_grad(set_to_none=True)
            g += 1
            dt = time.perf_counter() - t0
            ss["seconds"] += dt
            append_jsonl(out / "train_log.jsonl", {"global_update": g, "stage": si + 1, "loss": loss_total / 16,
                                                   "lr_connector": lrs.get("connector_decay"), "lr_lora": lrs.get("lora_decay"),
                                                   "grad_norm": gn, "skipped": scaler.get_scale() < sc, "seconds": dt,
                                                   "kinds": dict(zip(*np.unique(kinds, return_counts=True)))})
            if g % 50 == 0:
                log(f"g {g:4d}/{total} | stage {si + 1} | loss {loss_total / 16:.4f} | lr_c {lrs.get('connector_decay'):.2e} | "
                    f"gnorm {gn:.2f} | {dt:.2f}s/upd | peak {gpu_peak()['peak_allocated_gb']:.1f} GB | draws {mix['draws']}")
            if g in devpts:
                t1 = time.perf_counter()
                d = dev_eval(vlm, data, feats, sel, dev)
                ss["dev_eval_seconds"] += time.perf_counter() - t1
                append_jsonl(out / "dev_log.jsonl", {"global_update": g, "stage": si + 1, **d})
                log(f"dev @{g}: " + ", ".join(f"{k}={v:.4f}" for k, v in d.items()))
            ss["peak_allocated_gb"] = max(ss["peak_allocated_gb"], gpu_peak()["peak_allocated_gb"])
            if g % SHARED["ckpt_every"] == 0 or g == s1:
                ck = save_ckpt(si, s0, opt, sched, rng, mix)
                append_jsonl(out / "checkpoints.jsonl", {"global_update": g, "stage": si + 1, "bytes": dir_size_bytes(ck)})
            elif time.time() - last_save["t"] >= args.safety_minutes * 60:
                save_ckpt(si, s0, opt, sched, rng, mix)
                log(f"safety checkpoint at global update {g}")
        ss["mix"] = mix
        if g >= s1 and not (out / f"weights_after_stage{si + 1}.pt").exists():
            torch.save(trainable_state(vlm), out / f"weights_after_stage{si + 1}.pt")
        write_json(out / "stage_stats.json", stage_stats)
    write_json(out / "run_summary.json", {"tokens": args.tokens, "global_updates": g, "examples_presented": g * 16,
                                          "wall_seconds_this_session": time.perf_counter() - t_session,
                                          "train_seconds": sum(v["seconds"] for v in stage_stats.values()),
                                          "dev_eval_seconds": sum(v["dev_eval_seconds"] for v in stage_stats.values())})
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("training finished")


if __name__ == "__main__":
    main()
