"""Task 3 Step 4: parallel sampling vs one longer response vs sequential refinement, live, on the 50 fixed MATH questions.

    python -m task3.t3_live generate --data_root DIR --out OUT [--resume_from FILE ...]
    python -m task3.t3_live score --out OUT

Every question gets five independently seeded calls, all on the same GPU, batch size 1, FP16, no quantization:
    single     one response, at most 3,072 new tokens
    parallel   three responses, at most 1,024 new tokens each (the first is also the refinement's initial response)
    revision   the first parallel response appended as an assistant message, then the review instruction; one
               critique-and-revision response of at most 2,048 new tokens
Seeds: 6304 + 10 x (question index) + call index (single 0, parallel 1-3, revision 4).
Each finished call is appended to OUT/live/live_calls.jsonl at once; a rerun skips calls already saved there or in
--resume_from files (an attached earlier output), so an interrupted session continues where it stopped.
Gold answers are used only for grading after generation, never to choose between the initial and revised answers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import SEED, Checks, append_jsonl, read_jsonl, write_json  # noqa: E402
from task3 import t3_data as D  # noqa: E402

MODEL_ID = os.environ.get("PA3_T3_MODEL", "Qwen/Qwen2.5-3B-Instruct")              # env override: offline tests only
REVISION = os.environ.get("PA3_T3_REVISION", "aa8e72537993ba99e69dfaafa59ed015b17504d1") or None
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
CALLS = [("single", 0, 3072), ("parallel_1", 1, 1024), ("parallel_2", 2, 1024), ("parallel_3", 3, 1024), ("revision", 4, 2048)]
SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0}
ALLOWANCE = 3072
N_BOOT = 1000


def live_questions(data_root):
    """The first 50 held-out MATH questions (same split rule as the offline experiments), with text and gold answer."""
    import pyarrow.parquet as pq
    folder = D.find_folders([data_root])["math"]
    t = pq.read_table(folder / "generations.parquet", columns=["sample_id", "rollout_id", "question_text", "gold_answer"]).to_pandas()
    t = t[t.rollout_id == 0].set_index("sample_id")
    sp = D.make_splits(t.index.tolist())
    return [{"q_index": i, "sample_id": q, "problem": t.loc[q, "question_text"], "gold": t.loc[q, "gold_answer"]}
            for i, q in enumerate(sp["live"])]


def messages(problem, initial=None):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}]
    if initial is not None:
        m += [{"role": "assistant", "content": initial}, {"role": "user", "content": REVIEW}]
    return m


def generate(model, tok, prompt, cap, seed, eos_ids):
    import torch
    from transformers import set_seed
    dev = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    set_seed(seed)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cap, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0],
                             eos_token_id=eos_ids, **SAMPLING)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    eos_end = bool(new) and new[-1] in eos_ids
    kept = new[:-1] if eos_end else new                     # the stop token is not counted (as in the supplied data)
    text = tok.decode(kept, skip_special_tokens=True)
    return {"input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": len(kept), "output_tokens_incl_stop": len(new),
            "stopped": eos_end, "truncated": (not eos_end) and len(new) >= cap, "response": text, "extracted": D.last_boxed(text),
            "seconds": sec, "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else 0.0}


def run_generate(data_root, out, resume_from=(), limit=None, stop_at=None, log=print):
    import torch
    from transformers import AutoTokenizer
    from common.utils import load_causal_lm
    ldir = Path(out) / "live"
    ldir.mkdir(parents=True, exist_ok=True)
    path = ldir / "live_calls.jsonl"
    done = {}
    for f in list(resume_from) + [path]:
        if Path(f).exists():
            for r in read_jsonl(f):
                done[(r["sample_id"], r["call"])] = r
    if done:                                                # carry an attached earlier output forward (union of all copies)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in done.values():
                f.write(json.dumps(r) + "\n")
        tmp.replace(path)
    qs = live_questions(data_root)[:limit]
    write_json(ldir / "live_questions.json", qs)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = {"revision": REVISION} if REVISION else {}
    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    t_load = time.time()
    model = load_causal_lm(MODEL_ID, torch.float16 if dev.type == "cuda" else torch.float32, **kw).to(dev).eval()
    eos = model.generation_config.eos_token_id
    eos = tok.eos_token_id if eos is None else eos
    eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if tok.convert_tokens_to_ids("<|im_end|>") not in eos_ids and tok.convert_tokens_to_ids("<|im_end|>") is not None:
        eos_ids.append(tok.convert_tokens_to_ids("<|im_end|>"))
    import importlib.metadata as md
    env = {"model": MODEL_ID, "revision": REVISION, "dtype": str(next(model.parameters()).dtype), "batch_size": 1,
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
           "transformers": md.version("transformers"), "sampling": SAMPLING, "eos_ids": eos_ids,
           "model_load_seconds_excluded": round(time.time() - t_load, 1)}
    write_json(ldir / "live_env.json", env)
    generate(model, tok, tok.apply_chat_template(messages("What is 1 + 1?"), tokenize=False, add_generation_prompt=True),
             8, SEED, eos_ids)                              # warm-up, not recorded
    t0 = time.time()
    for q in qs:
        if stop_at and time.time() > stop_at:
            log("  time limit reached: stopping cleanly; run again with this output attached to continue")
            write_json(ldir / "PAUSED", {"time": time.time()})
            return path
        for call, k, cap in CALLS:
            if (q["sample_id"], call) in done:
                continue
            if call == "revision":
                init = done[(q["sample_id"], "parallel_1")]["response"]
                prompt = tok.apply_chat_template(messages(q["problem"], init), tokenize=False, add_generation_prompt=True)
            else:
                prompt = tok.apply_chat_template(messages(q["problem"]), tokenize=False, add_generation_prompt=True)
            seed = SEED + 10 * q["q_index"] + k
            r = {"sample_id": q["sample_id"], "q_index": q["q_index"], "call": call, "seed": seed, "cap": cap, "prompt": prompt,
                 **generate(model, tok, prompt, cap, seed, eos_ids)}
            append_jsonl(path, r)
            done[(q["sample_id"], call)] = r
        n_done = sum(1 for x in qs if all((x["sample_id"], c) in done for c, _, _ in CALLS))
        log(f"  {n_done} / {len(qs)} questions finished ({(time.time() - t0) / 60:.1f} min this session)")
    return path


# ---------------------------------------------------------------------------- scoring
def _vote(keys):
    """Self-consistency over the three parallel answers; ties -> earliest response."""
    cnt = {}
    for i, k in enumerate(keys):
        cnt.setdefault(k, [0, i])[0] += 1
    return min(cnt.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[1][1]


def bootstrap_diff(a, b, n=N_BOOT, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def score(out, checks=None):
    C = checks or Checks()
    ldir = Path(out) / "live"
    calls = {(r["sample_id"], r["call"]): r for r in read_jsonl(ldir / "live_calls.jsonl")}
    qs = json.load(open(ldir / "live_questions.json"))
    complete = [q for q in qs if all((q["sample_id"], c) in calls for c, _, _ in CALLS)]
    C.check(len(complete) == len(qs), "every live question has all five calls", len(qs), len(complete))
    rows = []
    for q in complete:
        g = {c: calls[(q["sample_id"], c)] for c, _, _ in CALLS}
        answers = [g["single"]["extracted"], g["parallel_1"]["extracted"], g["parallel_2"]["extracted"],
                   g["parallel_3"]["extracted"], g["revision"]["extracted"]]
        keys, corr = D.math_groups_and_grades(answers, q["gold"])
        pk = keys[1:4]
        win = _vote(pk)
        rows.append({
            "sample_id": q["sample_id"], "q_index": q["q_index"], "gold": q["gold"],
            "single_answer": answers[0], "single_correct": corr[0],
            "parallel_answers": answers[1:4], "parallel_answer": answers[1 + win], "parallel_correct": corr[1 + win],
            "parallel_no_answer": pk[win] == D.NO_ANSWER, "parallel_votes": max(pk.count(k) for k in pk),
            "initial_answer": answers[1], "initial_correct": corr[1], "revised_answer": answers[4], "revised_correct": corr[4],
            "revision_changed_answer": keys[4] != keys[1],
            **{f"{s}_{m}": v for s, cs in [("single", ["single"]), ("parallel", ["parallel_1", "parallel_2", "parallel_3"]),
                                            ("refinement", ["parallel_1", "revision"])]
               for m, v in [("input_tokens", sum(g[c]["input_tokens"] for c in cs)),
                            ("output_tokens", sum(g[c]["output_tokens"] for c in cs)),
                            ("seconds", sum(g[c]["seconds"] for c in cs)),
                            ("truncated_calls", sum(g[c]["truncated"] for c in cs)),
                            ("unextracted_calls", sum(g[c]["extracted"] is None for c in cs)),
                            ("peak_gpu_gb", max(g[c]["peak_gpu_gb"] for c in cs))]},
            "revision_input_tokens": g["revision"]["input_tokens"], "revision_output_tokens": g["revision"]["output_tokens"],
            "revision_truncated": g["revision"]["truncated"], "initial_truncated": g["parallel_1"]["truncated"]})
    pq_ = pd.DataFrame(rows)
    pq_.to_csv(ldir / "live_per_question.csv", index=False)
    n_calls = {"single": 1, "parallel": 3, "refinement": 2}
    summ = []
    for s in ["single", "parallel", "refinement"]:
        acc = pq_[f"{s}_correct" if s != "refinement" else "revised_correct"]
        final_missing = {"single": pq_.single_answer.isna(), "parallel": pq_.parallel_no_answer,
                         "refinement": pq_.revised_answer.isna()}[s]
        summ.append({"strategy": {"single": "single longer response (3,072)", "parallel": "parallel x3 (1,024) + self-consistency",
                                  "refinement": "initial (1,024) + critique-and-revision (2,048)"}[s],
                     "accuracy": float(acc.mean()), "questions": len(pq_),
                     "mean input tokens": float(pq_[f"{s}_input_tokens"].mean()),
                     "mean output tokens": float(pq_[f"{s}_output_tokens"].mean()),
                     "unused output allowance (tokens)": float(ALLOWANCE - pq_[f"{s}_output_tokens"].mean()),
                     "total runtime (min)": float(pq_[f"{s}_seconds"].sum() / 60),
                     "mean runtime per question (s)": float(pq_[f"{s}_seconds"].mean()),
                     "truncated calls (%)": float(100 * pq_[f"{s}_truncated_calls"].sum() / (n_calls[s] * len(pq_))),
                     "calls without an extracted answer (%)": float(100 * pq_[f"{s}_unextracted_calls"].sum() / (n_calls[s] * len(pq_))),
                     "final answer missing (%)": float(100 * final_missing.mean()),
                     "peak GPU memory (GB)": float(pq_[f"{s}_peak_gpu_gb"].max())})
    summ = pd.DataFrame(summ)
    summ.to_csv(ldir / "live_summary.csv", index=False)
    boot = []
    for a, b in [("parallel_correct", "single_correct"), ("revised_correct", "single_correct"), ("revised_correct", "parallel_correct"),
                 ("revised_correct", "initial_correct")]:
        est, lo, hi = bootstrap_diff(pq_[a], pq_[b])
        boot.append({"comparison": f"{a.split('_')[0]} - {b.split('_')[0]}", "diff_pp": 100 * est, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi})
    pd.DataFrame(boot).to_csv(ldir / "live_bootstrap.csv", index=False)
    tr = {"wrong -> correct": int((~pq_.initial_correct & pq_.revised_correct).sum()),
          "correct -> wrong": int((pq_.initial_correct & ~pq_.revised_correct).sum()),
          "correct -> correct": int((pq_.initial_correct & pq_.revised_correct).sum()),
          "wrong -> wrong": int((~pq_.initial_correct & ~pq_.revised_correct).sum()),
          "revision changed the answer group": int(pq_.revision_changed_answer.sum()),
          "initial response truncated": int(pq_.initial_truncated.sum()), "revision truncated": int(pq_.revision_truncated.sum()),
          "mean revision prompt tokens": float(pq_.revision_input_tokens.mean())}
    write_json(ldir / "live_transitions.json", tr)
    C.observe(f"live: strategies ({len(pq_)} MATH questions)", summ.round(3).to_dict(orient="records"))
    C.observe("live: revision transitions relative to the shared initial answer", tr)
    C.observe("live: paired bootstrap accuracy differences (pp)", pd.DataFrame(boot).round(2).to_dict(orient="records"))
    return summ, tr, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "score", "all"])
    ap.add_argument("--data_root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/t3")
    ap.add_argument("--resume_from", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=None, help="only the first K questions (tests)")
    ap.add_argument("--stop_at", type=float, default=None, help="epoch seconds: start no new question after this time")
    a = ap.parse_args()
    C = Checks()
    if a.stage in ("generate", "all"):
        run_generate(a.data_root, a.out, a.resume_from, a.limit, a.stop_at)
    if a.stage in ("score", "all"):
        score(a.out, C)
    C.summary()
    C.save(Path(a.out) / "live" / "t3_live_checks.json")


if __name__ == "__main__":
    main()
