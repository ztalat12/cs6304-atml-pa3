"""Task 2 final evaluation: both final-test suites under four image conditions.

    correct        the question's own image
    complementary  the partner image of its pair (whose reference answer is DIFFERENT)
    mismatched     an unrelated image from another pair of the same suite (fixed mapping, saved in the selection file)
    no_image       visual tokens removed, text unchanged (shorter prefix; mask and positions rebuilt)
The question and the ORIGINAL references stay fixed in every condition.
Greedy decoding, thinking disabled, 16 new tokens, batch size 8.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.lora import inject_lora  # noqa: E402
from common.utils import SEED, gpu_peak, write_json  # noqa: E402
from task2.t2_data import vqa_normalize, vqa_soft_accuracy  # noqa: E402
from task2.t2_features import FeatureStore  # noqa: E402
from task2.t2_model import LM_ID, VLM, Connector, SeqBuilder, collate, load_lm  # noqa: E402
from task2.t2_train import SHARED  # noqa: E402

CONDS = ["correct", "complementary", "mismatched", "no_image"]
N_BOOT = 4000


def load_final(run_dir, dev):
    lm = load_lm(dev)
    vlm = VLM(lm, Connector(d_out=lm.config.hidden_size).to(dev))
    inject_lora(vlm.lm, SHARED["lora_targets"], SHARED["lora_r"], SHARED["lora_alpha"], 0.0, seed=SEED, log=lambda *a: None)
    sd = torch.load(Path(run_dir) / "weights_after_stage3.pt", map_location="cpu")
    res = vlm.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "lora_" in k or k.startswith("connector")]
    assert not bad and not res.unexpected_keys, (bad, res.unexpected_keys)
    vlm.eval()
    return vlm, len(sd)


def questions(sel, suite):
    rows = []
    mis = sel[f"{suite}_mismatch"]
    for p, pr in enumerate(sel[suite]):
        for j, q in enumerate(pr):
            other = pr[1 - j]
            rows.append({"pair": p, "slot": j, "type": q["type"], "question": q["question"], "image": q["image"],
                         "partner_image": other["image"], "mismatch_image": mis[p][j],
                         "canonical": vqa_normalize(q["consensus"] if "consensus" in q else q["answer"]),
                         "reference": q["target"] if "target" in q else q["answer"],
                         "partner_canonical": vqa_normalize(other["consensus"] if "consensus" in other else other["answer"]),
                         "answers": q.get("answers")})
    return rows


def image_for(row, cond):
    return {"correct": row["image"], "complementary": row["partner_image"], "mismatched": row["mismatch_image"],
            "no_image": None}[cond]


@torch.no_grad()
def run_condition(vlm, builder, feats, rows, cond, dev, n_vis, bs=8):
    """Greedy answers + teacher-forced NLL of the ORIGINAL reference (termination excluded), batch size 8."""
    tok, im_end = builder.tok, builder.im_end
    preds, nlls, t_gen = [], [], 0.0
    nv = 0 if cond == "no_image" else n_vis
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        f = feats.get([image_for(r, cond) for r in chunk]) if nv else None
        b = {k: v.to(dev) for k, v in collate([builder.build("qa", nv, question=r["question"]) for r in chunk],
                                               builder.pad, left_pad=True).items()}
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            gen = vlm.greedy(b, f, max_new=16, eos_ids=(im_end, tok.eos_token_id))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_gen += time.perf_counter() - t0
        for row in gen.tolist():
            cut = row[:row.index(im_end)] if im_end in row else row
            preds.append(tok.decode(cut, skip_special_tokens=True).strip())
        bt = {k: v.to(dev) for k, v in collate([builder.build("qa", nv, question=r["question"], target=r["reference"]) for r in chunk],
                                                builder.pad).items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            nlls += vlm.example_nll(bt, f, exclude_last=True).cpu().tolist()
    return preds, nlls, t_gen


def score(rows, preds, suite):
    out = []
    for r, p in zip(rows, preds):
        pn = vqa_normalize(p)
        acc = vqa_soft_accuracy(pn, [vqa_normalize(a) for a in r["answers"]]) if suite.startswith("vqa") else float(pn == r["canonical"])
        out.append({"pred": p, "pred_norm": pn, "acc": acc, "exact_canonical": pn == r["canonical"],
                    "follows_partner": pn == r["partner_canonical"], "empty": pn == ""})
    return out


def pair_level(rows, scored, nll):
    P = {}
    for r, s, n in zip(rows, scored, nll):
        d = P.setdefault(r["pair"], {"acc": [], "exact": [], "pred": [], "nll": [], "type": r["type"], "follow": []})
        d["acc"].append(s["acc"])
        d["exact"].append(s["exact_canonical"])
        d["pred"].append(s["pred_norm"])
        d["nll"].append(n)
        d["follow"].append(s["follows_partner"])
    return {k: {"acc": float(np.mean(v["acc"])), "pair_correct": float(all(v["exact"])), "identical": float(v["pred"][0] == v["pred"][1]),
                "nll": float(np.mean(v["nll"])), "type": v["type"], "follow": float(np.mean(v["follow"]))} for k, v in P.items()}


def bootstrap_mean(a, n=N_BOOT, seed=SEED):
    a = np.asarray(a, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    m = a[idx].mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bootstrap_diff(a, b, n=N_BOOT, seed=SEED):
    """95% interval for mean(a - b) over pairs; the SAME resampled pair indices are used for both conditions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=196)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel = json.load(open(args.sel))
    feats = FeatureStore(args.feats, pool49=args.tokens == 49)
    tok = AutoTokenizer.from_pretrained(LM_ID)
    builder = SeqBuilder(tok)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    vlm, n_loaded = load_final(args.run, dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, per_pair_all = {"loaded_tensors": n_loaded}, {}
    for suite in ["grounding_test", "vqa_test"]:
        rows = questions(sel, suite)
        per_pair, scored_by = {}, {}
        for cond in CONDS:
            preds, nll, tg = run_condition(vlm, builder, feats, rows, cond, dev, args.tokens)
            sc = score(rows, preds, suite)
            scored_by[cond] = sc
            with open(out / f"pred_{suite}_{cond}.jsonl", "w") as f:
                for r, s, n in zip(rows, sc, nll):
                    f.write(json.dumps({**{k: v for k, v in r.items() if k != "answers"}, "condition": cond,
                                        "image_used": image_for(r, cond), **s, "ref_nll": n}) + "\n")
            pp = pair_level(rows, sc, nll)
            per_pair[cond] = pp
            keys = sorted(pp)
            types = sorted({r["type"] for r in rows})
            m = {"accuracy": float(np.mean([s["acc"] for s in sc])),
                 "accuracy_ci95": bootstrap_mean([pp[k]["acc"] for k in keys])[1:],
                 "accuracy_by_type": {t: float(np.mean([s["acc"] for r, s in zip(rows, sc) if r["type"] == t])) for t in types},
                 "pair_accuracy": float(np.mean([pp[k]["pair_correct"] for k in keys])),
                 "identical_answer_rate": float(np.mean([pp[k]["identical"] for k in keys])),
                 "empty_answer_rate": float(np.mean([s["empty"] for s in sc])),
                 "follows_partner_answer_rate": float(np.mean([s["follows_partner"] for s in sc])),
                 "changed_from_correct_image_rate": float(np.mean([s["pred_norm"] != c["pred_norm"]
                                                                   for s, c in zip(sc, scored_by["correct"])])),
                 "mean_reference_nll": float(np.mean(nll)),
                 "gen_seconds_per_answer": tg / len(rows), "n_questions": len(rows), "n_pairs": len(keys)}
            summary[f"{suite}|{cond}"] = m
            print(f"{suite:15s} {cond:13s} acc {m['accuracy']:.3f} | pair {m['pair_accuracy']:.3f} | identical "
                  f"{m['identical_answer_rate']:.3f} | follows-partner {m['follows_partner_answer_rate']:.3f} | changed "
                  f"{m['changed_from_correct_image_rate']:.3f} | NLL {m['mean_reference_nll']:.3f} | empty {m['empty_answer_rate']:.3f}",
                  flush=True)
        keys = sorted(per_pair["correct"])
        adv = {}
        for other in ["complementary", "mismatched", "no_image"]:
            adv[f"acc_correct_minus_{other}"] = bootstrap_diff([per_pair["correct"][k]["acc"] for k in keys],
                                                               [per_pair[other][k]["acc"] for k in keys])
            adv[f"nll_{other}_minus_correct"] = bootstrap_diff([per_pair[other][k]["nll"] for k in keys],
                                                               [per_pair["correct"][k]["nll"] for k in keys])
        for t in sorted({per_pair["correct"][k]["type"] for k in keys}):
            kt = [k for k in keys if per_pair["correct"][k]["type"] == t]
            adv[f"acc_correct_minus_complementary|{t}"] = bootstrap_diff([per_pair["correct"][k]["acc"] for k in kt],
                                                                         [per_pair["complementary"][k]["acc"] for k in kt])
            adv[f"nll_complementary_minus_correct|{t}"] = bootstrap_diff([per_pair["complementary"][k]["nll"] for k in kt],
                                                                         [per_pair["correct"][k]["nll"] for k in kt])
        summary[f"{suite}|advantages"] = adv
        per_pair_all[suite] = {c: {str(k): v for k, v in pp.items()} for c, pp in per_pair.items()}
        for k, v in adv.items():
            print(f"   {k:48s} {v[0]:+.3f}  [{v[1]:+.3f}, {v[2]:+.3f}]")
    summary["resources"] = {**gpu_peak(), "tokens": args.tokens, "bootstrap_resamples": N_BOOT}
    write_json(out / "final_metrics.json", summary)
    write_json(out / "per_pair.json", per_pair_all)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
