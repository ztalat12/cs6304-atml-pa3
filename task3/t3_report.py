"""Task 3 tables and figures, built only from saved results (OUT/analysis and OUT/live), so they can be rebuilt without
the supplied data:  python -m task3.t3_report --results results/task3/t3 --out results/task3/t3_report_rebuilt"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3.t3_methods import methods_for  # noqa: E402

TITLE = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
VLM = {"math": False, "mathvista": True}
STYLE = {"FirstAns": ("k", ":"), "SC": ("C0", "-"), "MaxLogProb": ("C1", "--"), "MaxSelfCert": ("C2", "--"),
         "SelfCert+RWC": ("C2", "-"), "Shortest": ("0.5", ":"), "MaxVision": ("C4", "--"), "Vision+RWC": ("C4", "-"),
         "MaxVisConf": ("C3", "--"), "VisConf+RWC": ("C3", "-"), "MaxVisConfTok": ("C5", "--"), "VisConfTok+RWC": ("C5", "-")}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def method_table(res, out, name, budgets=(4, 32)):
    """Compact comparison: accuracy and cost at the chosen budgets, with paired bootstrap differences vs self-consistency."""
    a = res / "analysis"
    s = pd.read_csv(a / f"{name}_summary.csv")
    b = pd.read_csv(a / f"{name}_bootstrap.csv")
    rows = []
    for m in methods_for(VLM[name]):
        r = {"method": m}
        for n in budgets:
            x = s[(s.method == m) & (s.N == n)].iloc[0]
            r[f"acc N={n} (%)"] = 100 * x.accuracy
            r[f"tokens N={n}"] = x.mean_token_cost
            bb = b[(b.method == m) & (b.N == n) & (b.vs == "SC")]
            r[f"vs SC N={n} (pp) [95% CI]"] = "" if bb.empty else \
                f"{bb.diff_pp.iloc[0]:+.1f} [{bb.ci_low_pp.iloc[0]:+.1f}, {bb.ci_high_pp.iloc[0]:+.1f}]"
        rows.append(r)
    t = pd.DataFrame(rows)
    cov = s[s.method == "FirstAns"].set_index("N").oracle_coverage
    t.loc[len(t)] = {"method": "oracle coverage (upper bound)", **{f"acc N={n} (%)": 100 * cov[n] for n in budgets}}
    t.to_csv(out / f"t3_{name}_methods.csv", index=False)
    return t


def decomposition_table(res, out, name):
    s = pd.read_csv(res / "analysis" / f"{name}_summary.csv")
    g = pd.read_csv(res / "analysis" / f"{name}_answer_groups.csv")
    keep = ["FirstAns", "SC", "SelfCert+RWC"] + (["VisConf+RWC"] if VLM[name] else [])
    t = s[s.method.isin(keep)][["method", "N", "accuracy", "missed_with_correct_available", "no_correct_candidate", "oracle_coverage"]].copy()
    for c in t.columns[2:]:
        t[c] = 100 * t[c]
    t = t.merge(g, on="N")
    t.to_csv(out / f"t3_{name}_decomposition.csv", index=False)
    return t


def curves_figure(res, out, names):
    plt = _plt()
    fig, ax = plt.subplots(1, 2 * len(names), figsize=(6.2 * len(names) * 2 / 1.6, 4.0))
    for j, name in enumerate(names):
        s = pd.read_csv(res / "analysis" / f"{name}_summary.csv")
        a1, a2 = ax[2 * j], ax[2 * j + 1]
        for m in methods_for(VLM[name]):
            d = s[s.method == m].sort_values("N")
            col, ls = STYLE[m]
            a1.plot(d.N, 100 * d.accuracy, ls, color=col, marker="o", ms=3, label=m)
            a2.plot(d.mean_token_cost, 100 * d.accuracy, ls, color=col, marker="o", ms=3, label=m)
        d = s[s.method == "FirstAns"].sort_values("N")
        a1.plot(d.N, 100 * d.oracle_coverage, "-", color="k", lw=2.2, label="oracle coverage")
        ad = res / "analysis" / f"{name}_adaptive_heldout.csv"
        if ad.exists():
            t = pd.read_csv(ad)
            for _, r in t[~t.policy.str.startswith("fixed")].iterrows():
                a2.scatter(r.mean_cost, 100 * r.accuracy, marker="*", s=140, zorder=5, label=r.policy.split(" (")[0] + " stopping")
        a1.set_xscale("log", base=2); a1.set_xticks([1, 2, 4, 8, 16, 32]); a1.set_xticklabels([1, 2, 4, 8, 16, 32])
        a1.set_xlabel("candidates N"); a1.set_ylabel("held-out accuracy (%)"); a1.set_title(f"{TITLE[name].split(' (')[0]}: accuracy and coverage vs N", fontsize=10)
        from matplotlib.ticker import FuncFormatter, NullFormatter
        a2.set_xscale("log"); a2.set_xlabel("mean generated tokens per question (log scale)"); a2.set_ylabel("held-out accuracy (%)")
        a2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}")); a2.xaxis.set_minor_formatter(NullFormatter())
        a2.set_title(f"{TITLE[name].split(' (')[0]}: accuracy vs tokens", fontsize=10)
        a1.grid(alpha=.3); a2.grid(alpha=.3)
    h, l = ax[-1].get_legend_handles_labels()
    h0, l0 = ax[-2].get_legend_handles_labels()
    seen = dict(zip(l0 + l, h0 + h))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=min(8, len(seen)), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "t3_accuracy_curves.png", dpi=150)
    plt.close(fig)


def decomposition_figure(res, out, names):
    plt = _plt()
    fig, ax = plt.subplots(1, len(names), figsize=(5.5 * len(names), 3.6), squeeze=False)
    for j, name in enumerate(names):
        s = pd.read_csv(res / "analysis" / f"{name}_summary.csv")
        meths = ["SC", "SelfCert+RWC"] + (["VisConf+RWC"] if VLM[name] else [])
        x, labels, w = [], [], 0.8 / len(meths)
        Ns = sorted(s.N.unique())
        for k, m in enumerate(meths):
            d = s[s.method == m].set_index("N").loc[Ns]
            pos = np.arange(len(Ns)) + (k - (len(meths) - 1) / 2) * w
            ax[0, j].bar(pos, 100 * d.accuracy, w, color="C2", edgecolor="k", lw=.3, label="selected correctly" if k == 0 else None)
            ax[0, j].bar(pos, 100 * d.missed_with_correct_available, w, bottom=100 * d.accuracy, color="C1", edgecolor="k", lw=.3,
                         label="missed: a correct candidate was available" if k == 0 else None)
            ax[0, j].bar(pos, 100 * d.no_correct_candidate, w, bottom=100 * (d.accuracy + d.missed_with_correct_available), color="0.75",
                         edgecolor="k", lw=.3, label="no correct candidate" if k == 0 else None)
            for p_ in pos:
                ax[0, j].text(p_, 101, m.replace("+RWC", "\n+RWC"), ha="center", va="bottom", fontsize=5, rotation=90)
        ax[0, j].set_xticks(np.arange(len(Ns))); ax[0, j].set_xticklabels(Ns); ax[0, j].set_xlabel("candidates N")
        ax[0, j].set_ylabel("% of held-out questions"); ax[0, j].set_ylim(0, 125); ax[0, j].set_title(TITLE[name])
    ax[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "t3_failure_decomposition.png", dpi=150)
    plt.close(fig)


def signals_figure(res, out):
    a = res / "analysis"
    if not (a / "mathvista_conf_vision_per_rollout.csv").exists():
        return
    plt = _plt()
    cv = pd.read_csv(a / "mathvista_conf_vision_per_rollout.csv")
    tr = json.load(open(a / "mathvista_trajectories.json"))
    nq = len(tr) // 2
    fig = plt.figure(figsize=(12, 2.6 * max(nq, 1)))
    gs = fig.add_gridspec(max(nq, 1), 3)
    ax0 = fig.add_subplot(gs[:, 0])
    med = cv.groupby("sample_id").corr_c_v.median().dropna()
    ax0.hist(med, bins=30, color="C3", alpha=.8)
    ax0.axvline(0, color="k", lw=1)
    ax0.set_xlabel("within-rollout corr(cᵢ,ₜ, vᵢ,ₜ): per-question median")
    ax0.set_ylabel("held-out questions"); ax0.set_title("Confidence–vision correlation")

    def smooth(x, k=16):
        x = np.asarray(x, float)
        return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) >= k else x

    for i in range(nq):
        for j in range(2):
            t = tr[2 * i + j]
            axx = fig.add_subplot(gs[i, 1 + j])
            axx.plot(smooth(t["c"]), color="C2", lw=1, label="calibrated confidence c")
            ax2 = axx.twinx()
            ax2.plot(smooth(t["v"]), color="C4", lw=1, label="visual share v")
            axx.set_ylim(0, 1); axx.set_ylabel("c", color="C2", fontsize=8); ax2.set_ylabel("v", color="C4", fontsize=8)
            axx.set_title(f"Q {t['sample_id']} · rollout {t['rollout']} · {'correct' if t['correct'] else 'incorrect'} "
                          f"(Z={t['Z']:.3f})", fontsize=8)
            axx.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
            if i == nq - 1:
                axx.set_xlabel("generated token (16-token moving average)", fontsize=8)
            if i == 0 and j == 1:
                h1, l1 = axx.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
                axx.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "t3_mathvista_signals.png", dpi=150)
    plt.close(fig)


def live_tables(res, out):
    d = res / "live"
    if not (d / "live_summary.csv").exists():
        return None
    t = pd.read_csv(d / "live_summary.csv")
    t.to_csv(out / "t3_live_summary.csv", index=False)
    return t


def build(res, out, log=print):
    res, out = Path(res), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in ["math", "mathvista"] if (res / "analysis" / f"{n}_summary.csv").exists()]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for n in names:
        log(f"\n==== {TITLE[n]}: method comparison (held-out)")
        log(method_table(res, out, n).round(1).to_string(index=False))
        decomposition_table(res, out, n)
        for f in ["signals", "adaptive_heldout"]:
            p = res / "analysis" / f"{n}_{f}.csv"
            if p.exists():
                pd.read_csv(p).to_csv(out / f"t3_{n}_{f}.csv", index=False)
    if names:
        curves_figure(res, out, names)
        decomposition_figure(res, out, names)
        signals_figure(res, out)
    lt = live_tables(res, out)
    if lt is not None:
        log("\n==== live experiment (50 MATH questions)\n" + lt.round(3).to_string(index=False))
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="the t3/ folder (with analysis/ and live/)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.results, a.out)
