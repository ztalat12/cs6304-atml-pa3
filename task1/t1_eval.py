"""Task 1 evaluation for the seven conditions (benchmarks + qualitative generations).

    baseline      unchanged Qwen3-0.6B
    full_mid      Full FT after 20,000 examples (update 625)
    full_final    Full FT after 40,000 examples
    l2sp_final    Full FT + L2-SP
    lora4_final   LoRA r=4, merged into a fresh FP16 backbone
    lora16_final  LoRA r=16, merged
    oplora_final  OPLoRA r=16 (projected factors), merged

Usage (one process per GPU):  python -m task1.t1_eval --conditions baseline full_mid --out /kaggle/working/t1_eval
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation (set before CUDA starts)
import torch  # noqa: E402
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.lora import attach_exported_adapter, merge_adapter_into  # noqa: E402
from common.utils import SEED, append_jsonl, gpu_peak, load_causal_lm, seed_everything, write_json  # noqa: E402
from task1.t1_data import MODEL_ID, build_prompt  # noqa: E402

CONDITIONS = {
    "baseline":     dict(method=None),
    "full_mid":     dict(method="full", export="export_mid"),
    "full_final":   dict(method="full", export="export_final"),
    "l2sp_final":   dict(method="l2sp", export="export_final"),
    "lora4_final":  dict(method="lora4", adapter="adapter_final_projected.pt", scale=8 / 4),
    "lora16_final": dict(method="lora16", adapter="adapter_final_projected.pt", scale=32 / 16),
    "oplora_final": dict(method="oplora", adapter="adapter_final_projected.pt", scale=32 / 16),
}
ORDER = list(CONDITIONS)
GEN_CAP = 1024
# Batch sizes to try, largest first; a CUDA out-of-memory error moves on to the next one (the size used is recorded).
# Likelihood scoring needs far more memory per example than generation: lm-eval sorts requests longest-first and holds
# log-probabilities over the whole 151,936-token vocabulary for every position of every sequence in the batch
# (batch 32 x ~900-token BoolQ passages x 151,936 x 2 bytes is ~9 GB, twice over for log_softmax).
GSM8K_BATCH_SIZES = (32, 16, 8)
MC_BATCH_SIZES = (8, 4, 2, 1)
QUAL_CAP = 512
MC_TASKS = ["hellaswag", "boolq", "arc_easy"]
EXPECTED_N = {"gsm8k": 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_run(method, roots=("/kaggle/working", "/kaggle/input")):
    """Locate a FINISHED t1_runs/<method>/: the working dir first; otherwise, among attached outputs, the copy
    that finished most recently (so an OPLoRA rerun wins over an older attached copy)."""
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        hits = [p.parent for p in root.rglob("final_summary.json")
                if p.parent.name == method and p.parent.parent.name == "t1_runs" and (p.parent / "DONE").exists()]
        if hits:
            return max(hits, key=lambda d: (d / "DONE").read_text().strip())
    return None


def load_hf(path_or_id, dtype=torch.float16):
    return load_causal_lm(path_or_id, dtype)


@torch.no_grad()
def merged_vs_unmerged(adapter, scale, tok, dev, prompts):
    """Logits of (fresh backbone + unmerged adapter) vs (backbone with the adapter merged in), same small batch."""
    base = load_hf(MODEL_ID).to(dev).eval()
    unmerged = attach_exported_adapter(base, adapter, scale, dropout=0.0).eval()
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        lu = unmerged(**enc).logits.float()
    del unmerged, base
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    merged = load_hf(MODEL_ID).to(dev).eval()
    merge_adapter_into(merged, adapter, scale, log=log)
    lm = merged(**enc).logits.float()
    m = enc["attention_mask"].bool()
    diff = (lu - lm).abs()[m]
    out = {"max_abs_logit_diff": diff.max().item(), "mean_abs_logit_diff": diff.mean().item(),
           "argmax_agreement": (lu.argmax(-1) == lm.argmax(-1))[m].float().mean().item(),
           "n_positions": int(m.sum())}
    return merged, out


def build_condition(cond, tok, dev, probe_prompts):
    spec = CONDITIONS[cond]
    info = {"condition": cond}
    if spec["method"] is None:
        model = load_hf(MODEL_ID)
        info["source"] = MODEL_ID
    else:
        run = find_run(spec["method"])
        if run is None:
            raise FileNotFoundError(f"t1_runs/{spec['method']} not found: attach the training output that contains it")
        info["run_dir"] = str(run)
        if "export" in spec:
            model = load_hf(run / spec["export"])
            info["source"] = str(run / spec["export"])
        else:
            adapter = torch.load(run / spec["adapter"], map_location="cpu")
            model, chk = merged_vs_unmerged(adapter, spec["scale"], tok, dev, probe_prompts)
            info["merge_check"] = chk
            info["source"] = str(run / spec["adapter"]) + " merged into fresh " + MODEL_ID
            log(f"merged-vs-unmerged logits: max |diff| {chk['max_abs_logit_diff']:.4f}, "
                f"mean {chk['mean_abs_logit_diff']:.5f}, argmax agreement {chk['argmax_agreement']:.4f}")
    model.to(dev).eval()
    model.config.use_cache = True
    return model, info


# ----------------------------------------------------------------------------- lm-eval with exact truncation flags
def make_tracking_lm(model, tok, batch_size):
    """An lm-eval HFLM that also records, for every generated continuation, how many tokens were produced
    and whether generation ended by itself or ran into the token cap. lm-eval only returns the final text,
    so without this the truncation flag would have to be guessed from re-tokenised text."""
    from lm_eval.models.huggingface import HFLM

    class TrackingHFLM(HFLM):
        gen_log: dict = {}

        def _model_generate(self, context, max_length, stop, **kw):
            out = super()._model_generate(context=context, max_length=max_length, stop=stop, **kw)
            mask = kw.get("attention_mask")
            L = context.shape[1]
            cap = max_length - L
            ends = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
            ge = getattr(self.model.generation_config, "eos_token_id", None)
            ends |= set(ge if isinstance(ge, (list, tuple)) else [ge] if ge is not None else [])
            for b in range(out.shape[0]):
                ctx_ids = context[b][mask[b].bool()] if mask is not None else context[b]
                key = self.tokenizer.decode(ctx_ids.tolist(), skip_special_tokens=False)
                gen = out[b, L:].tolist()
                n, hit = len(gen), False
                for i, t in enumerate(gen):
                    if t in ends:
                        n, hit = i, True
                        break
                self.gen_log[key] = {"n_raw": n, "hit_eos": hit, "cap": cap, "stop": list(stop),
                                     "raw_text": self.tokenizer.decode(gen[:n], skip_special_tokens=True)}
            return out

    lm = TrackingHFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, enable_thinking=False, max_length=4096)
    lm.gen_log = {}
    return lm


def _context(sample):
    """The prompt string of a logged lm-eval sample (works for tuple- and dict-style 'arguments')."""
    a = sample["arguments"][0]
    if isinstance(a, dict):
        a = list(a.values())
    return a[0] if isinstance(a, (list, tuple)) else a


def gsm8k_rows(samples, tok, gen_log, cap=GEN_CAP):
    """One row per GSM8K question: response, both extractions, correctness, generated tokens and flags."""
    by_doc = {}
    for s in samples:
        d = by_doc.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "question": s["doc"]["question"],
                                            "target": s["target"], "response": s["resps"][0][0],
                                            "prompt": _context(s)})
        tag = "flex" if s["filter"] == "flexible-extract" else "strict"
        d[f"{tag}_extracted"] = s["filtered_resps"][0]
        d[f"{tag}_correct"] = float(s["exact_match"])
    rows, n_tracked = [], 0
    for d in sorted(by_doc.values(), key=lambda x: x["doc_id"]):
        info = gen_log.get(d["prompt"])
        if info is not None:
            n_tracked += 1
            cut_by_stop = any(t and t in info["raw_text"] for t in info["stop"])
            d["n_gen_tokens"] = len(tok(d["response"], add_special_tokens=False)["input_ids"]) if cut_by_stop else info["n_raw"]
            d["stop_reason"] = "stop_string" if cut_by_stop else ("end_of_turn" if info["hit_eos"] else "token_cap")
            d["truncated_at_cap"] = (not cut_by_stop) and (not info["hit_eos"]) and info["n_raw"] >= info["cap"]
        else:          # fallback: estimate from the re-tokenised text
            n = len(tok(d["response"], add_special_tokens=False)["input_ids"])
            d["n_gen_tokens"], d["stop_reason"] = n, "unknown"
            d["truncated_at_cap"] = n >= cap - 8
        d["unextractable"] = d.get("flex_extracted") in ("[invalid]", None, "")
        rows.append(d)
    return rows, n_tracked


def _with_oom_fallback(fn, sizes, what):
    """Run fn(batch_size) with each size in turn until one fits in GPU memory. Returns (result, size used)."""
    last = None
    for bs in sizes:
        oom = False
        try:
            return fn(bs), bs
        except torch.cuda.OutOfMemoryError as e:     # noqa: PERF203
            last = repr(e)[:200]
            oom = True
        if oom:                                        # outside the except block, so the traceback's tensors can be freed
            log(f"{what}: out of GPU memory at batch size {bs}; retrying with a smaller batch")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"{what}: out of memory at every batch size {sizes}: {last}")


SEEDS = dict(random_seed=SEED, numpy_random_seed=SEED, torch_random_seed=SEED, fewshot_random_seed=SEED)


def run_gsm8k(model, tok, out_dir, sizes=GSM8K_BATCH_SIZES):
    """GSM8K (generation). Saves gsm8k_samples.jsonl + gsm8k_part.json, so a later crash never loses this stage."""
    from lm_eval import simple_evaluate

    def attempt(bs):
        lm = make_tracking_lm(model, tok, bs)
        g = simple_evaluate(model=lm, tasks=["gsm8k_cot"], num_fewshot=0, apply_chat_template=True,
                            gen_kwargs={"max_gen_toks": GEN_CAP}, log_samples=True, bootstrap_iters=1000, **SEEDS)
        return g, lm.gen_log

    t0 = time.perf_counter()
    (g, gen_log), bs = _with_oom_fallback(attempt, sizes, "GSM8K")
    part = {"gsm8k_seconds": time.perf_counter() - t0, "gsm8k_batch_size": bs}
    gr = g["results"]["gsm8k_cot"]
    part["gsm8k_flexible_em"] = gr["exact_match,flexible-extract"]
    part["gsm8k_flexible_em_stderr"] = gr.get("exact_match_stderr,flexible-extract")
    part["gsm8k_strict_em"] = gr["exact_match,strict-match"]
    part["gsm8k_strict_em_stderr"] = gr.get("exact_match_stderr,strict-match")
    part["versions_gsm8k"] = g.get("versions", {})
    gcfg = g["configs"]["gsm8k_cot"]
    part["lm_eval_config"] = {"gsm8k": {k: gcfg.get(k) for k in ["num_fewshot", "generation_kwargs", "filter_list"]}}
    rows, n_tracked = gsm8k_rows(g["samples"]["gsm8k_cot"], tok, gen_log)
    with open(out_dir / "gsm8k_samples.jsonl", "w") as f:
        for d in rows:
            f.write(json.dumps(d) + "\n")
    n = len(rows)
    part["gsm8k_n"] = n
    part["gsm8k_tracked_generations"] = n_tracked
    part["gsm8k_truncation_rate"] = sum(d["truncated_at_cap"] for d in rows) / n
    part["gsm8k_unextractable_rate"] = sum(d["unextractable"] for d in rows) / n
    part["gsm8k_mean_gen_tokens"] = sum(d["n_gen_tokens"] for d in rows) / n
    part["gsm8k_stop_reasons"] = {k: sum(d["stop_reason"] == k for d in rows)
                                  for k in ["end_of_turn", "stop_string", "token_cap", "unknown"]}
    not_trunc = [d for d in rows if not d["truncated_at_cap"]]
    part["gsm8k_correct_given_not_truncated"] = sum(d["flex_correct"] for d in not_trunc) / max(1, len(not_trunc))
    part["first_gsm8k_prompt"] = rows[0]["prompt"]
    write_json(out_dir / "gsm8k_part.json", part)
    return part


def run_mc(model, tok, out_dir, sizes=MC_BATCH_SIZES):
    """HellaSwag / BoolQ / ARC-Easy (candidate likelihood). Saves per-task samples + mc_part.json."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    def attempt(bs):
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, max_length=4096)
        return simple_evaluate(model=lm, tasks=MC_TASKS, num_fewshot=0, apply_chat_template=False,
                               log_samples=True, bootstrap_iters=1000, **SEEDS)

    t0 = time.perf_counter()
    o, bs = _with_oom_fallback(attempt, sizes, "HellaSwag/BoolQ/ARC-Easy")
    part = {"mc_seconds": time.perf_counter() - t0, "mc_batch_size": bs, "versions_mc": o.get("versions", {}), "mc_n": {}}
    for t in MC_TASKS:
        r = o["results"][t]
        part[f"{t}_acc"] = r["acc,none"]
        part[f"{t}_acc_stderr"] = r.get("acc_stderr,none")
        if "acc_norm,none" in r:
            part[f"{t}_acc_norm"] = r["acc_norm,none"]
            part[f"{t}_acc_norm_stderr"] = r.get("acc_norm_stderr,none")
        with open(out_dir / f"{t}_samples.jsonl", "w") as f:
            for smp in o["samples"][t]:
                f.write(json.dumps({"doc_id": smp["doc_id"], "target": smp["target"], "acc": smp.get("acc"),
                                    "acc_norm": smp.get("acc_norm"), "context": _context(smp),
                                    "loglik": [r_[0][0] if isinstance(r_[0], (list, tuple)) else r_[0] for r_ in smp["resps"]]},
                                   default=str) + "\n")
        part["mc_n"][t] = len(o["samples"][t])
    write_json(out_dir / "mc_part.json", part)
    return part


def combine(gpart, mpart):
    """metrics.json = both stages together (same keys as before, plus the batch sizes used)."""
    agg = {k: v for k, v in gpart.items() if not k.startswith("versions")}
    agg.update({k: v for k, v in mpart.items() if not k.startswith("versions") and k != "mc_n"})
    agg["versions"] = {**gpart.get("versions_gsm8k", {}), **mpart.get("versions_mc", {})}
    agg["n_examples"] = {"gsm8k": gpart["gsm8k_n"], **mpart["mc_n"]}
    return agg


# ----------------------------------------------------------------------------- qualitative
def load_qual_prompts(path):
    qs = yaml.safe_load(open(path))
    out = []
    for q in qs:
        base = re.sub(r"_(json|template)$", "", q["id"])
        fmt = "json" if q["id"].endswith("_json") else "template" if q["id"].endswith("_template") else "plain"
        out.append({"id": q["id"], "question": base, "format": fmt, "system": q.get("system"),
                    "user": q["paraphrases"][0]})
    return out


@torch.no_grad()
def qualitative(model, tok, dev, qpath, out_path, cond):
    """One greedy response per prompt, thinking disabled, 512-new-token cap. Written to a .part file and renamed only
    when all 24 are done, so an interrupted run never leaves a partial file that looks finished."""
    prompts = load_qual_prompts(qpath)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    out_path = Path(out_path)
    part = out_path.with_suffix(".jsonl.part")
    part.unlink(missing_ok=True)
    for p in prompts:
        text = build_prompt(tok, p["user"], system=p["system"])
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=QUAL_CAP, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=[im_end, tok.eos_token_id])
        new = gen[0, enc["input_ids"].shape[1]:].tolist()
        hit_eos = len(new) > 0 and new[-1] in (im_end, tok.eos_token_id)
        resp = tok.decode(new, skip_special_tokens=True)
        append_jsonl(part, {"condition": cond, **p, "prompt_text": text, "response": resp,
                            "n_tokens": len(new) - int(hit_eos), "truncated": (not hit_eos) and len(new) >= QUAL_CAP})
    os.replace(part, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=ORDER)
    ap.add_argument("--out", default="/kaggle/working/t1_eval")
    ap.add_argument("--qual_yaml", default=str(Path(__file__).with_name("first_plot_questions.yaml")))
    ap.add_argument("--probe", default=None, help="t1_probe_set.json (prompts for the merge-check batch)")
    ap.add_argument("--gsm8k_batch_sizes", type=int, nargs="+", default=list(GSM8K_BATCH_SIZES))
    ap.add_argument("--mc_batch_sizes", type=int, nargs="+", default=list(MC_BATCH_SIZES))
    ap.add_argument("--skip_benchmarks", action="store_true")
    ap.add_argument("--skip_qual", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    seed_everything(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    probe_prompts = [build_prompt(tok, "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many "
                                       "clips in May. How many clips did Natalia sell altogether in April and May?\nA:"),
                     "Roof shingle removal: A man is sitting on a roof. He"]
    if args.probe:
        pj = json.load(open(args.probe))
        probe_prompts = [build_prompt(tok, x["text"]) for x in pj["gsm8k"][:4]] + [x["text"] for x in pj["hellaswag"][:4]]
    for cond in args.conditions:
        out = Path(args.out) / cond
        out.mkdir(parents=True, exist_ok=True)
        need_bench = not args.skip_benchmarks and not (out / "metrics.json").exists()
        need_qual = not args.skip_qual and not (out / "qual_responses.jsonl").exists()
        if not need_bench and not need_qual:
            log(f"{cond}: already evaluated, skipping")
            continue
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model, info = build_condition(cond, tok, dev, probe_prompts)
        info["build_seconds"] = time.perf_counter() - t0
        if need_bench:
            if (out / "gsm8k_part.json").exists():
                gpart = json.load(open(out / "gsm8k_part.json"))
                log(f"{cond}: GSM8K already done (reused)")
            else:
                log(f"{cond}: GSM8K (generation, 1,319 questions)")
                gpart = run_gsm8k(model, tok, out, tuple(args.gsm8k_batch_sizes))
                log(f"{cond}: GSM8K {gpart['gsm8k_flexible_em']:.4f} (batch {gpart['gsm8k_batch_size']}, "
                    f"{gpart['gsm8k_seconds'] / 60:.1f} min)")
            if (out / "mc_part.json").exists():
                mpart = json.load(open(out / "mc_part.json"))
                log(f"{cond}: HellaSwag/BoolQ/ARC-Easy already done (reused)")
            else:
                log(f"{cond}: HellaSwag / BoolQ / ARC-Easy (candidate likelihood)")
                mpart = run_mc(model, tok, out, tuple(args.mc_batch_sizes))
            agg = combine(gpart, mpart)
            write_json(out / "metrics.json", {**info, **agg, **gpu_peak()})
            log(f"{cond}: GSM8K {agg['gsm8k_flexible_em']:.4f} | HellaSwag {agg['hellaswag_acc_norm']:.4f} | "
                f"BoolQ {agg['boolq_acc']:.4f} | ARC-E {agg['arc_easy_acc_norm']:.4f} | "
                f"truncated {agg['gsm8k_truncation_rate']:.3f} | unextractable {agg['gsm8k_unextractable_rate']:.3f} | "
                f"batch sizes {agg['gsm8k_batch_size']} / {agg['mc_batch_size']}")
        if need_qual:
            log(f"{cond}: qualitative generations (24 prompts, greedy, {QUAL_CAP} tokens)")
            t0 = time.perf_counter()
            qualitative(model, tok, dev, args.qual_yaml, out / "qual_responses.jsonl", cond)
            info["qual_seconds"] = time.perf_counter() - t0
        write_json(out / "eval_meta.json", {**info, **gpu_peak()})
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
