"""Task 2 figures and tables, built only from saved files (train_log, dev_log, stage_stats, final_metrics, predictions)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import read_json, read_jsonl  # noqa: E402
from task2.t2_train import stage_bounds  # noqa: E402

SUITES = {"grounding_test": "Annotation-derived", "vqa_test": "Natural VQAv2"}
CONDS = ["correct", "complementary", "mismatched", "no_image"]


def learning_curves(run, out, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = pd.DataFrame(read_jsonl(Path(run) / "train_log.jsonl"))
    dv = pd.DataFrame(read_jsonl(Path(run) / "dev_log.jsonl"))
    ends = [b for _, b in stage_bounds()]
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    ax[0].plot(tr.global_update, tr.loss, alpha=0.25, lw=0.6, label="training loss (per update)")
    ax[0].plot(tr.global_update, tr.loss.ewm(alpha=0.03).mean(), lw=1.5, label="training loss (EMA)")
    ax[0].plot(dv.global_update, dv.caption_dev_loss, "o-", label="development caption loss")
    ax[0].set(title="Training loss & development caption loss", xlabel="update", ylabel="mean token NLL")
    for j, suite in enumerate(["grounding_dev", "vqa_dev"]):
        a = ax[1 + j]
        for cond, style in [("correct", "o-"), ("complementary", "s--"), ("no_image", "^:")]:
            a.plot(dv.global_update, dv[f"{suite}_nll_{cond}"], style, label=cond.replace("_", " ") + " image" if cond != "no_image" else "no image")
        a.set(title=f"{'Annotation-derived' if j == 0 else 'Natural VQAv2'} dev: reference-answer NLL", xlabel="update", ylabel="NLL")
    ax[3].plot(dv.global_update, dv.grounding_dev_swap_penalty, "o-", label="annotation-derived")
    ax[3].plot(dv.global_update, dv.vqa_dev_swap_penalty, "s-", label="natural VQAv2")
    ax[3].axhline(0, color="grey", lw=0.8)
    ax[3].set(title="Image-swap penalty ΔNLL (partner − correct)", xlabel="update", ylabel="ΔNLL")
    for a in ax:
        for e in ends[:-1]:
            a.axvline(e, color="k", lw=0.6, ls="--")
        a.legend(fontsize=7)
    fig.suptitle("Dashed vertical lines: stage boundaries (2,250 and 3,750)", fontsize=8, y=0.02)
    fig.tight_layout()
    fig.savefig(Path(out) / f"t2_learning_curves{tag}.png", dpi=200)
    dv.to_csv(Path(out) / f"t2_dev_metrics{tag}.csv", index=False)
    return tr, dv


def final_tables(eval_dir, out, tag=""):
    m = read_json(Path(eval_dir) / "final_metrics.json")
    rows = []
    for suite, nm in SUITES.items():
        for c in CONDS:
            r = m[f"{suite}|{c}"]
            row = {"Suite": nm, "Condition": c, "Acc": 100 * r["accuracy"],
                   "Acc 95% CI": f"[{100 * r['accuracy_ci95'][0]:.1f}, {100 * r['accuracy_ci95'][1]:.1f}]",
                   "Pair acc": 100 * r["pair_accuracy"], "Identical": 100 * r["identical_answer_rate"],
                   "Changed vs correct img": 100 * r["changed_from_correct_image_rate"],
                   "Follows partner": 100 * r["follows_partner_answer_rate"], "Empty": 100 * r["empty_answer_rate"],
                   "Ref NLL": r["mean_reference_nll"], "ms/answer": 1000 * r["gen_seconds_per_answer"]}
            for t, v in r["accuracy_by_type"].items():
                row[f"Acc[{t}]"] = 100 * v
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / f"t2_final_conditions{tag}.csv", index=False)
    arows = []
    for suite, nm in SUITES.items():
        for k, (d, lo, hi) in m[f"{suite}|advantages"].items():
            scale = 100 if k.startswith("acc") else 1
            arows.append({"Suite": nm, "Quantity": k, "Unit": "pp" if scale == 100 else "nats", "Estimate": scale * d,
                          "CI low": scale * lo, "CI high": scale * hi, "Excludes 0": (lo > 0) or (hi < 0)})
    adf = pd.DataFrame(arows)
    adf.to_csv(Path(out) / f"t2_advantages{tag}.csv", index=False)
    return df, adf, m


def select_examples(eval_dir, n_per=2):
    """Stated selection rule: for each suite and question type, walk the pairs in saved order and keep the first n_per
    pairs whose two correct-image predictions DIFFER (image-sensitive) and the first n_per whose two predictions are
    IDENTICAL (image-insensitive). The rule uses predictions only, never which answer is right."""
    picks = []
    for suite in SUITES:
        rows = {c: read_jsonl(Path(eval_dir) / f"pred_{suite}_{c}.jsonl") for c in CONDS}
        by_pair = {}
        for c in CONDS:
            for r in rows[c]:
                by_pair.setdefault(r["pair"], {}).setdefault(c, [None, None])[r["slot"]] = r
        for t in sorted({r["type"] for r in rows["correct"]}):
            diff, same = [], []
            for p in sorted(by_pair):
                q = by_pair[p]["correct"]
                if q[0]["type"] != t:
                    continue
                (diff if q[0]["pred_norm"] != q[1]["pred_norm"] else same).append(p)
            for kind, lst in [("image-sensitive", diff[:n_per]), ("image-insensitive", same[:n_per])]:
                for p in lst:
                    picks.append({"suite": suite, "type": t, "kind": kind, "pair": p, "rows": by_pair[p]})
    return picks


def example_table(picks):
    out = []
    for e in picks:
        for slot in [0, 1]:
            r = e["rows"]["correct"][slot]
            out.append({"suite": SUITES[e["suite"]], "type": e["type"], "kind": e["kind"], "pair": e["pair"], "slot": slot,
                        "image": r["image"], "question": r["question"], "reference": r["reference"],
                        "pred_correct_image": r["pred"], "pred_partner_image": e["rows"]["complementary"][slot]["pred"],
                        "pred_mismatched_image": e["rows"]["mismatched"][slot]["pred"], "pred_no_image": e["rows"]["no_image"][slot]["pred"]})
    return pd.DataFrame(out)


def example_figure(picks, image_paths, out, max_pairs=8, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    picks = picks[:max_pairs]
    if not picks:
        return
    fig, ax = plt.subplots(len(picks), 2, figsize=(7.5, 3.2 * len(picks)))
    ax = np.atleast_2d(ax)
    for i, e in enumerate(picks):
        for slot in [0, 1]:
            r = e["rows"]["correct"][slot]
            a = ax[i, slot]
            p = image_paths.get(r["image"])
            if p:
                a.imshow(Image.open(p).convert("RGB").resize((224, 224)))
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"Q: {r['question']}\nref: {r['reference']} | pred: {r['pred']} | swap: "
                        f"{e['rows']['complementary'][slot]['pred']} | no-img: {e['rows']['no_image'][slot]['pred']}", fontsize=7)
        ax[i, 0].set_ylabel(f"{SUITES[e['suite']]}\n{e['type']} · {e['kind']}", fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out) / f"t2_examples{tag}.png", dpi=160)


def resource_table(run, feat_stats, eval_dir):
    st = read_json(Path(run) / "stage_stats.json")
    rows = [{"Item": "SigLIP feature caching (once, shared by both runs)", "Seconds": feat_stats.get("seconds"),
             "Peak GPU (GB)": feat_stats.get("peak_allocated_gb"), "Trainable params": 0,
             "Notes": f"{feat_stats.get('n_images')} images, {feat_stats.get('bytes', 0) / 1e9:.2f} GB FP16"}]
    for k in sorted(st):
        s = st[k]
        rows.append({"Item": f"Stage {k}: {s['name']}", "Seconds": s["seconds"], "Peak GPU (GB)": s.get("peak_allocated_gb"),
                     "Trainable params": s["trainable_params"],
                     "Notes": f"connector {s['connector_params']:,} + LoRA {s['lora_params']:,}; development evaluation "
                              f"{s.get('dev_eval_seconds', 0):.0f} s (not included)"})
    m = read_json(Path(eval_dir) / "final_metrics.json")
    gen = [v["gen_seconds_per_answer"] for k, v in m.items() if "|" in k and "advantages" not in k]
    rows.append({"Item": "Final-test generation (greedy, ≤16 tokens, batch 8)", "Seconds": float(np.mean(gen)),
                 "Peak GPU (GB)": m["resources"]["peak_allocated_gb"], "Trainable params": 0, "Notes": "seconds per answer"})
    return pd.DataFrame(rows)


def compression_table(m196, m49):
    """196 vs 49 visual tokens, final test, correct image + advantages (optional experiment)."""
    rows = []
    for suite, nm in SUITES.items():
        for tag, m in [("196", m196), ("49", m49)]:
            r = m[f"{suite}|correct"]
            a = m[f"{suite}|advantages"]["acc_correct_minus_complementary"]
            rows.append({"Suite": nm, "Visual tokens": tag, "Acc (correct img)": 100 * r["accuracy"],
                         **{f"Acc[{t}]": 100 * v for t, v in r["accuracy_by_type"].items()},
                         "Pair acc": 100 * r["pair_accuracy"], "Identical": 100 * r["identical_answer_rate"],
                         "Acc advantage vs swap (pp)": 100 * a[0], "95% CI": f"[{100 * a[1]:.1f}, {100 * a[2]:.1f}]",
                         "ms/answer": 1000 * r["gen_seconds_per_answer"]})
    return pd.DataFrame(rows)
