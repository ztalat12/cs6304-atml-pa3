"""Task 2: extra analyses computed from the saved evaluation files only (no GPU, no model, nothing re-generated).

    python -m task2.t2_extra --results results/task2/t2 --out results/task2/t2_extra

1. swap_breakdown   Under the complementary swap, does each answer stay the same, change to the partner image's answer,
                    or change to an answer that neither image's reference supports? (Manual, Step 5.)
2. prior_agreement  How often the correct-image answer equals the no-image answer (answers the image did not change),
                    and how often yes/no-type questions are answered "yes" with and without the image.
3. compare_49       196 vs 49 visual tokens on the same pairs: paired bootstrap (4,000 resamples of complete pairs,
                    seed 6304, the same routine as the final evaluation) for the difference in correct-image accuracy
                    and in the correct-minus-swap accuracy advantage.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2.t2_eval import bootstrap_diff  # noqa: E402

SUITES = {"grounding_test": "Annotation-derived", "vqa_test": "Natural VQAv2"}
CONDS = ["correct", "complementary", "mismatched", "no_image"]


def load(eval_dir, suite):
    """{condition: [row, ...]} in saved order (pair, slot)."""
    out = {}
    for c in CONDS:
        with open(Path(eval_dir) / f"pred_{suite}_{c}.jsonl") as f:
            out[c] = [json.loads(l) for l in f if l.strip()]
    keys = [[(r["pair"], r["slot"]) for r in out[c]] for c in CONDS]
    assert all(k == keys[0] for k in keys), "conditions are not aligned question by question"
    return out


def swap_breakdown(P, suite):
    rows = []
    groups = defaultdict(list)
    for c, s in zip(P["correct"], P["complementary"]):
        if s["pred_norm"] == c["pred_norm"]:
            kind = "unchanged"
        elif s["pred_norm"] == s["partner_canonical"]:
            kind = "changed to the partner image's answer"
        else:
            kind = "changed to another answer"
        groups["all"].append(kind)
        groups[c["type"]].append(kind)
    for t, ks in groups.items():
        n = Counter(ks)
        rows.append({"Suite": SUITES[suite], "Type": t, "Questions": len(ks),
                     **{k: 100 * n[k] / len(ks) for k in ["unchanged", "changed to the partner image's answer",
                                                          "changed to another answer"]}})
    return rows


def prior_agreement(P, suite):
    rows = []
    by = defaultdict(list)
    for c, n in zip(P["correct"], P["no_image"]):
        by["all"].append(c)
        by[c["type"]].append(c)
    noimg = {(r["pair"], r["slot"]): r["pred_norm"] for r in P["no_image"]}
    for t, rs in by.items():
        same = np.mean([r["pred_norm"] == noimg[(r["pair"], r["slot"])] for r in rs])
        row = {"Suite": SUITES[suite], "Type": t, "Questions": len(rs), "Correct-image answer = no-image answer (%)": 100 * same}
        if t in ("presence", "yes/no"):
            row["Reference 'yes' (%)"] = 100 * np.mean([r["canonical"] == "yes" for r in rs])
            row["Predicted 'yes', correct image (%)"] = 100 * np.mean([r["pred_norm"] == "yes" for r in rs])
            row["Predicted 'yes', no image (%)"] = 100 * np.mean([noimg[(r["pair"], r["slot"])] == "yes" for r in rs])
        if t != "all":
            top = Counter(noimg[(r["pair"], r["slot"])] for r in rs).most_common(3)
            row["Most common no-image answers"] = ", ".join(f"{a!r} x{k}" for a, k in top)
        rows.append(row)
    return rows


def per_pair(rows, value):
    d = defaultdict(list)
    for r in rows:
        d[r["pair"]].append(value(r))
    return {k: float(np.mean(v)) for k, v in d.items()}


def compare_49(P196, P49, suite):
    rows = []
    types = ["all"] + sorted({r["type"] for r in P196["correct"]})
    for t in types:
        keep = (lambda r: True) if t == "all" else (lambda r, t=t: r["type"] == t)
        acc = {n: {c: per_pair([r for r in P[c] if keep(r)], lambda r: r["acc"]) for c in ["correct", "complementary"]}
               for n, P in [(196, P196), (49, P49)]}
        keys = sorted(acc[196]["correct"])
        a196 = [acc[196]["correct"][k] for k in keys]
        a49 = [acc[49]["correct"][k] for k in keys]
        adv196 = [acc[196]["correct"][k] - acc[196]["complementary"][k] for k in keys]
        adv49 = [acc[49]["correct"][k] - acc[49]["complementary"][k] for k in keys]
        for q, a, b in [("correct-image accuracy", a196, a49), ("accuracy advantage over the swap", adv196, adv49)]:
            est, lo, hi = bootstrap_diff(a, b)
            rows.append({"Suite": SUITES[suite], "Type": t, "Pairs": len(keys), "Quantity": q,
                         "196 tokens": 100 * np.mean(a), "49 tokens": 100 * np.mean(b),
                         "196 − 49 (pp)": 100 * est, "95% CI low": 100 * lo, "95% CI high": 100 * hi,
                         "CI excludes 0": bool(lo > 0 or hi < 0)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="the unzipped t2/ folder from t2_results.zip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sw, pr, cmp = [], [], []
    for suite in SUITES:
        P196 = load(res / "eval196", suite)
        sw += swap_breakdown(P196, suite)
        pr += prior_agreement(P196, suite)
        if (res / "eval49").exists():
            cmp += compare_49(P196, load(res / "eval49", suite), suite)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    for name, rows in [("t2_swap_breakdown", sw), ("t2_prior_agreement", pr), ("t2_compare_49", cmp)]:
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out / f"{name}.csv", index=False)
            print(f"\n{name}\n" + df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
