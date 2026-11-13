"""Task 3 offline experiments on the supplied rollouts (Steps 1, 2, 3, 5 and the data for Step 6). CPU only.

    python -m task3.t3_offline prep    --data_root DIR --out OUT [--mathvista_meta JSON]
    python -m task3.t3_offline analyze --out OUT

prep     loads and checks both folders, grades every rollout, writes OUT/prep/ (per-rollout table, VLM token arrays,
         split IDs, random orders, scorer versions).
analyze  runs every selection rule at N in {1, 2, 4, 8, 16, 32} over the ten orders, the Step 3 signal analysis, the
         Step 5 stopping controllers (thresholds chosen on development, frozen, then held-out), and writes OUT/analysis/.
All reported numbers use the 800 held-out evaluation questions; the 200 development questions only choose thresholds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import SEED, Checks, read_json, write_json  # noqa: E402
from task3 import t3_data as D  # noqa: E402
from task3.t3_methods import QuestionPool, methods_for, select_all, stop_stats  # noqa: E402

DATASETS = {"math": False, "mathvista": True}          # name -> is VLM
CHECKPOINTS = (4, 8, 16)
TAUS = (0.5, 0.625, 0.75, 0.875, 1.0)
GAMMAS = (0.0, 0.1, 0.25, 0.5)
N_BOOT = 1000
QUALITY_SCORE = {"math": "self-certainty", "mathvista": "VisConf"}
MAIN_QUALITY_METHOD = {"math": "SelfCert+RWC", "mathvista": "VisConf+RWC"}


# ============================================================================ prep
def prep(data_root, out, mathvista_meta=None, datasets=("math", "mathvista"), checks=None):
    C = checks or Checks()
    out = Path(out)
    pdir = out / "prep"
    pdir.mkdir(parents=True, exist_ok=True)
    folders = D.find_folders([data_root])
    import importlib.metadata as md
    scorer = {"math-verify": md.version("math-verify"), "mathvista_rules": "MathVista evaluation/calculate_score.py "
              "normalize_extracted_answer, ported (task3/t3_data.py); multiple choice as option letters", "no_answer": D.NO_ANSWER}
    for name in datasets:
        C.check(name in folders, f"{name}: supplied folder found ({D.FOLDERS[name]}/)", D.FOLDERS[name], folders.get(name))
        if name not in folders:
            continue
        t0 = time.time()
        df, tok = D.load_folder(folders[name], DATASETS[name], C, name)
        D.readme_totals_check(df, name, C)
        meta = None
        if name == "mathvista":
            meta, src = D.load_mathvista_meta(out / "hf_cache", mathvista_meta)
            write_json(pdir / "mathvista_meta.json", {"meta": meta, "source": src})
            scorer["mathvista_metadata"] = src
            missing = sorted(set(df.sample_id) - set(meta))
            C.check(not missing, "mathvista: every question has released metadata (question/answer type, choices, precision)",
                    0, f"{len(missing)} missing, e.g. {missing[:3]}")
        pdf, info = D.prepare(df, name, meta)
        keep = ["dataset", "sample_id", "rollout_id", "T", "L", "K", "hit_max_new_tokens", "extracted_answer", "gold_answer",
                "group", "correct", "no_answer"] + (["V"] if DATASETS[name] else [])
        pdf[keep].to_parquet(pdir / f"{name}_rollouts.parquet", index=False)
        if DATASETS[name]:
            np.savez(pdir / f"{name}_tokens.npz", kl=tok["kl"], v=tok["v"], offsets=tok["offsets"])
        ids = sorted(pdf.sample_id.unique())
        sp = D.make_splits(ids)
        write_json(pdir / f"{name}_splits.json", sp)
        orders = D.make_orders(sp["sorted"])
        np.savez(pdir / f"{name}_orders.npz", ids=np.array(sp["sorted"], dtype=object), orders=np.stack([orders[q] for q in sp["sorted"]]))
        C.check(len(sp["dev"]) == D.N_DEV and len(sp["eval"]) == D.N_QUESTIONS - D.N_DEV and not set(sp["dev"]) & set(sp["eval"]),
                f"{name}: split = {D.N_DEV} development + {D.N_QUESTIONS - D.N_DEV} held-out questions, disjoint",
                (D.N_DEV, D.N_QUESTIONS - D.N_DEV), (len(sp["dev"]), len(sp["eval"])))
        C.observe(f"{name}: first five development / held-out IDs (sorted, then default_rng(6304) permutation)",
                  (sp["dev"][:5], sp["eval"][:5]))
        if name == "math":
            C.observe("math: the 50 live-generation questions (first 50 held-out)", sp["live"][:10] + ["..."])
        g = pdf.groupby("sample_id")
        C.observe(f"{name}: answer extraction / grading", {
            "rollouts correct (%)": round(100 * pdf.correct.mean(), 2), "NO_ANSWER (%)": round(100 * pdf.no_answer.mean(), 2),
            "questions with >= 1 correct rollout (%)": round(100 * g.correct.any().mean(), 2),
            "mean distinct answer groups per question (32 rollouts)": round(g.group.nunique().mean(), 2), **info})
        C.observe(f"{name}: preparation time", f"{time.time() - t0:.0f} s")
    write_json(pdir / "scorer.json", scorer)
    C.observe("scorer (fixed before evaluation)", scorer)
    return C


# ============================================================================ analysis helpers
def load_prepared(out, name):
    pdir = Path(out) / "prep"
    df = pd.read_parquet(pdir / f"{name}_rollouts.parquet")
    tok = None
    if DATASETS[name]:
        # read each array ONCE: every access to an NpzFile entry re-reads the whole array from disk, so indexing
        # tok["kl"] inside the question loop made 1,000 full copies (tens of GB) and the process was killed
        with np.load(pdir / f"{name}_tokens.npz") as z:
            tok = {"kl": z["kl"], "v": z["v"], "offsets": z["offsets"]}
    sp = read_json(pdir / f"{name}_splits.json")
    o = np.load(pdir / f"{name}_orders.npz", allow_pickle=True)
    orders = dict(zip(o["ids"].tolist(), o["orders"]))
    pools, row0 = {}, {}
    qs = df.sample_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    for s in starts:
        sub = df.iloc[s:s + D.N_ROLLOUTS]
        assert (sub.rollout_id.to_numpy() == np.arange(D.N_ROLLOUTS)).all()
        kw = {}
        if tok is not None:
            off = tok["offsets"][s:s + D.N_ROLLOUTS + 1]
            kw = {"V": sub.V.to_numpy(), "kl": tok["kl"][off[0]:off[-1]], "v": tok["v"][off[0]:off[-1]], "offsets": off - off[0]}
        pools[qs[s]] = QuestionPool(sub["T"], sub["L"], sub["K"], sub["group"].tolist(), sub["correct"], sub["no_answer"], **kw)
        row0[qs[s]] = s
    return df, pools, sp, orders, row0


def bootstrap_diff(a, b, n=N_BOOT, seed=SEED):
    """Mean of a - b over questions and a 95% interval from `n` paired resamples of questions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_selection(pools, orders, vlm, log=print):
    """Every rule at every budget for every order. One row per (question, order, N, method)."""
    meths = methods_for(vlm)
    rows, cov = [], []
    t0 = time.time()
    for qi, (q, pool) in enumerate(pools.items()):
        for o, perm in enumerate(orders[q]):
            for n in D.BUDGETS:
                idx = perm[:n]
                sel = select_all(pool, idx, vlm)
                tot = int(pool.T[idx].sum())
                for m in meths:
                    r = sel[m]
                    rows.append((q, o, n, m, r, bool(pool.correct[r]), int(pool.T[idx[0]]) if m == "FirstAns" else tot,
                                 pool.group_names[pool.g[r]]))
                gi = pool.g[idx]
                cov.append((q, o, n, bool(pool.correct[idx].any()), len(np.unique(gi)), len(np.unique(gi[~pool.no_answer[idx]])),
                            int(pool.no_answer[idx].sum())))
        if (qi + 1) % 200 == 0:
            log(f"  selection: {qi + 1} / {len(pools)} questions ({time.time() - t0:.0f} s)")
    sel = pd.DataFrame(rows, columns=["sample_id", "order", "N", "method", "selected_rollout", "correct", "token_cost", "selected_group"])
    cov = pd.DataFrame(cov, columns=["sample_id", "order", "N", "coverage", "n_groups", "n_groups_excl_no_answer", "n_no_answer"])
    return sel, cov


def summarise_selection(sel, cov, eval_ids):
    """Held-out summary: average over orders within a question, then over questions."""
    s = sel[sel.sample_id.isin(eval_ids)].merge(cov[["sample_id", "order", "N", "coverage"]], on=["sample_id", "order", "N"])
    s["missed"] = s.coverage & ~s.correct            # a correct candidate was available but not selected
    s["none"] = ~s.coverage                           # no correct candidate among the N
    perq = s.groupby(["method", "N", "sample_id"])[["correct", "token_cost", "missed", "none", "coverage"]].mean().reset_index()
    summ = perq.groupby(["method", "N"])[["correct", "token_cost", "missed", "none", "coverage"]].mean().reset_index()
    summ = summ.rename(columns={"correct": "accuracy", "token_cost": "mean_token_cost", "missed": "missed_with_correct_available",
                                "none": "no_correct_candidate", "coverage": "oracle_coverage"})
    c = cov[cov.sample_id.isin(eval_ids)].groupby(["N", "sample_id"])[["n_groups", "n_groups_excl_no_answer", "n_no_answer"]].mean()
    cg = c.groupby("N").mean().reset_index()
    return perq, summ, cg


def bootstrap_table(perq, methods, budgets):
    rows = []
    for n in budgets:
        p = perq[perq.N == n].pivot(index="sample_id", columns="method", values="correct")
        for m in methods:
            for ref in ["SC", "FirstAns"]:
                if m == ref:
                    continue
                est, lo, hi = bootstrap_diff(p[m], p[ref])
                rows.append({"N": n, "method": m, "vs": ref, "diff_pp": 100 * est, "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi,
                             "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


# ============================================================================ Step 3
def _spearman(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def quartile_gap(score, correct):
    """Correct rate among rollouts with score >= 75th percentile minus among score <= 25th percentile (within question)."""
    s = np.asarray(score, float)
    ok = ~np.isnan(s)
    s, c = s[ok], np.asarray(correct, float)[ok]
    if len(s) < 4:
        return np.nan
    lo, hi = np.percentile(s, [25, 75])
    return float(c[s >= hi].mean() - c[s <= lo].mean())


def signal_analysis(pools, eval_ids, vlm):
    """N = 32 (all rollouts): within-question associations of length and scores with correctness, confidence-vision
    correlation within rollouts, and tokenwise vs rollout-level fusion."""
    per_q, cv_rows = [], []
    for q in eval_ids:
        p = pools[q]
        idx = np.arange(D.N_ROLLOUTS)
        sc = {"length T": p.T.astype(float), "mean log-prob L": p.L, "self-certainty K": p.K}
        if vlm:
            Z, Zt, Cc, (q05, q95) = p.visconf(idx)
            sc.update({"visual engagement V": p.V, "calibrated confidence C": Cc, "VisConf Z = V*C": Z, "tokenwise Z": Zt})
            for r in idx:
                a, b = p.off[r], p.off[r + 1]
                if b - a < 3:
                    continue
                c = np.clip((p.kl[a:b] - q05) / max(q95 - q05, 1e-8), 0, 1)
                v = np.nan_to_num(p.v[a:b].astype(float), nan=0.0)
                corr = float(np.corrcoef(c, v)[0, 1]) if np.ptp(c) > 0 and np.ptp(v) > 0 else np.nan
                cv_rows.append({"sample_id": q, "rollout": int(r), "corr_c_v": corr, "cov_c_v": float(Zt[r] - Z[r]),
                                "correct": bool(p.correct[r]), "T": int(p.T[r])})
        mixed = 0 < p.correct.sum() < D.N_ROLLOUTS
        row = {"sample_id": q, "mixed": mixed, "frac_correct": float(p.correct.mean())}
        for k, s in sc.items():
            row[f"gap|{k}"] = quartile_gap(s, p.correct)
            row[f"rho_correct|{k}"] = _spearman(s, p.correct) if mixed else np.nan
            if k != "length T":
                row[f"rho_length|{k}"] = _spearman(p.T, s)
        per_q.append(row)
    per_q = pd.DataFrame(per_q)
    names = [c.split("|", 1)[1] for c in per_q.columns if c.startswith("gap|")]
    summ = []
    for k in names:
        m = per_q[per_q.mixed]
        summ.append({"score": k,
                     "top - bottom quartile correct rate, all questions (pp)": 100 * per_q[f"gap|{k}"].mean(),
                     "top - bottom quartile correct rate, mixed questions (pp)": 100 * m[f"gap|{k}"].mean(),
                     "Spearman with correctness, mixed questions (mean)": m[f"rho_correct|{k}"].mean(),
                     "Spearman with length (mean)": per_q[f"rho_length|{k}"].mean() if f"rho_length|{k}" in per_q else np.nan,
                     "mixed questions": int(per_q.mixed.sum())})
    cv = pd.DataFrame(cv_rows)
    cvs = {}
    if len(cv):
        perq_med = cv.groupby("sample_id").corr_c_v.median()
        cvs = {"rollouts with a defined correlation": int(cv.corr_c_v.notna().sum()),
               "per-question median correlation: median across questions": float(perq_med.median()),
               "per-question median correlation: 25th-75th percentile": [float(perq_med.quantile(.25)), float(perq_med.quantile(.75))],
               "questions whose median correlation is positive (%)": float(100 * (perq_med > 0).mean()),
               "rollouts with positive correlation (%)": float(100 * (cv.corr_c_v > 0).mean()),
               "mean correlation, correct vs incorrect rollouts": [float(cv[cv.correct].corr_c_v.mean()), float(cv[~cv.correct].corr_c_v.mean())],
               "tokenwise Z - Z = Cov(c, v): mean": float(cv.cov_c_v.mean()),
               "tokenwise Z - Z: share positive (%)": float(100 * (cv.cov_c_v > 0).mean())}
    return per_q, pd.DataFrame(summ), cv, cvs


def illustrative_trajectories(pools, eval_ids, n_q=3):
    """Stated rule: the first n_q held-out MathVista questions (split order) having both a correct and an incorrect rollout
    of at least MIN_TRAJ tokens; in each, the correct and the incorrect such rollout with the smallest rollout ID.
    Token-level c (calibrated on the question's 32 rollouts) and v are saved for plotting."""
    MIN_TRAJ = 32
    out = []
    for q in eval_ids:
        p = pools[q]
        long_ = p.T >= MIN_TRAJ
        if not ((p.correct & long_).any() and (~p.correct & long_).any()):
            continue
        _, _, Cc, (q05, q95) = p.visconf(np.arange(D.N_ROLLOUTS))
        Z = p.V * Cc
        for want in (True, False):
            r = int(np.flatnonzero((p.correct == want) & long_)[0])
            a, b = p.off[r], p.off[r + 1]
            c = np.clip((p.kl[a:b] - q05) / max(q95 - q05, 1e-8), 0, 1)
            out.append({"sample_id": q, "rollout": r, "correct": want, "answer_group": p.group_names[p.g[r]], "T": int(p.T[r]),
                        "V": float(p.V[r]), "C": float(Cc[r]), "Z": float(Z[r]), "K": float(p.K[r]),
                        "c": np.round(c, 4).tolist(), "v": np.round(np.nan_to_num(p.v[a:b].astype(float)), 5).tolist()})
        if len(out) >= 2 * n_q:
            break
    return out


# ============================================================================ Step 5
def stopping_table(pools, orders, ids, vlm):
    """Per (question, order): checkpoint statistics at n = 4, 8, 16 and the fixed-32 self-consistency result."""
    rows = []
    for q in ids:
        p = pools[q]
        for o, perm in enumerate(orders[q]):
            r = {"sample_id": q, "order": o}
            for n in CHECKPOINTS:
                st = stop_stats(p, perm[:n], vlm)
                r.update({f"{k}@{n}": v for k, v in st.items()})
                r[f"correct@{n}"] = bool(p.correct[st["sc_sel"]])
            st = stop_stats(p, perm, vlm)
            r.update({"correct@32": bool(p.correct[st["sc_sel"]]), "cost@32": st["cost"], "sc_sel@32": st["sc_sel"],
                      "frac_correct_pool": float(p.correct.mean()), "top_share@32": st["share"]})
            rows.append(r)
    return pd.DataFrame(rows)


def simulate(tab, family, tau, gamma):
    """Stop at the first checkpoint whose condition holds; otherwise use all 32. Returns per-row (stop n, correct, cost)."""
    stop = np.full(len(tab), 32)
    corr = tab["correct@32"].to_numpy().copy()
    cost = tab["cost@32"].to_numpy().copy()
    done = np.zeros(len(tab), bool)
    if tau is not None:
        for n in CHECKPOINTS:
            cond = tab[f"unique@{n}"].to_numpy() & ~tab[f"leader_no_answer@{n}"].to_numpy() & \
                (tab[f"share@{n}"].to_numpy() * n >= tau * n - 1e-9)
            if family == "quality":
                cond &= tab[f"agree@{n}"].to_numpy() & (tab[f"margin@{n}"].to_numpy() >= gamma - 1e-12)
            new = cond & ~done
            stop[new], corr[new], cost[new] = n, tab[f"correct@{n}"].to_numpy()[new], tab[f"cost@{n}"].to_numpy()[new]
            done |= new
    return stop, corr, cost


def evaluate_policy(tab, family, tau, gamma):
    stop, corr, cost = simulate(tab, family, tau, gamma)
    d = pd.DataFrame({"sample_id": tab.sample_id, "stop": stop, "correct": corr, "cost": cost})
    pq_ = d.groupby("sample_id")[["correct", "cost", "stop"]].mean()
    return {"accuracy": float(pq_.correct.mean()), "mean_cost": float(pq_.cost.mean()), "mean_rollouts": float(pq_.stop.mean()),
            **{f"stop@{n} (%)": float(100 * (stop == n).mean()) for n in (*CHECKPOINTS, 32)}}, pq_


def choose_settings(dev_tab):
    """For each controller family: the lowest-cost setting whose development accuracy is at most 1 pp below fixed-32
    self-consistency (ties: higher accuracy, then smaller tau, then smaller gamma); always-32 is the fallback."""
    base, _ = evaluate_policy(dev_tab, "agreement", None, None)
    grid = []
    for fam in ["agreement", "quality"]:
        grid.append({"family": fam, "tau": None, "gamma": None, **base, "eligible": True})
        for tau in TAUS:
            for gamma in ([None] if fam == "agreement" else GAMMAS):
                r, _ = evaluate_policy(dev_tab, fam, tau, gamma or 0.0)
                grid.append({"family": fam, "tau": tau, "gamma": gamma, **r,
                             "eligible": r["accuracy"] >= base["accuracy"] - 0.01 - 1e-12})
    grid = pd.DataFrame(grid)
    chosen = {}
    for fam in ["agreement", "quality"]:
        g = grid[(grid.family == fam) & grid.eligible].copy()
        g["_tau"] = g.tau.fillna(9.0)
        g["_gamma"] = g.gamma.fillna(9.0)
        best = g.sort_values(["mean_cost", "accuracy", "_tau", "_gamma"], ascending=[True, False, True, True]).iloc[0]
        chosen[fam] = {"tau": None if pd.isna(best.tau) else float(best.tau),
                       "gamma": None if pd.isna(best.gamma) else float(best.gamma),
                       "dev_accuracy": float(best.accuracy), "dev_mean_cost": float(best.mean_cost),
                       "dev_fixed32_accuracy": base["accuracy"], "dev_fixed32_cost": base["mean_cost"]}
    return grid, chosen


def adaptive_heldout(eval_tab, chosen, summ_sel):
    rows, perq = [], {}
    base, b_pq = evaluate_policy(eval_tab, "agreement", None, None)
    rows.append({"policy": "fixed-32 self-consistency", **base})
    for fam, ch in chosen.items():
        r, pq_ = evaluate_policy(eval_tab, fam, ch["tau"], ch["gamma"] or 0.0)
        est, lo, hi = bootstrap_diff(pq_.correct, b_pq.correct)
        label = f"{fam}: fallback, always 32 (no cheaper setting within 1 pp on development)" if ch["tau"] is None else \
            f"{fam} (tau={ch['tau']}" + (f", gamma={ch['gamma']})" if fam == "quality" else ")")
        rows.append({"policy": label, **r,
                     "accuracy - fixed32 (pp)": 100 * est, "CI low": 100 * lo, "CI high": 100 * hi})
        perq[fam] = pq_
    sc = summ_sel[summ_sel.method == "SC"]
    for _, s in sc.iterrows():
        rows.append({"policy": f"fixed-{int(s.N)} self-consistency", "accuracy": s.accuracy, "mean_cost": s.mean_token_cost,
                     "mean_rollouts": float(s.N)})
    return pd.DataFrame(rows), perq


def early_stop_analysis(eval_tab, chosen):
    """Incorrect early stops, and stopping behaviour against the (retrospective) fraction of correct rollouts."""
    out = {}
    bins = [-0.001, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0]
    labels = ["0", "(0, .25]", "(.25, .5]", "(.5, .75]", "(.75, 1)", "1"]
    for fam, ch in chosen.items():
        stop, corr, cost = simulate(eval_tab, fam, ch["tau"], ch["gamma"] or 0.0)
        early = stop < 32
        wrong_early = early & ~corr
        rec = wrong_early & eval_tab["correct@32"].to_numpy()
        t = pd.DataFrame({"bin": pd.cut(eval_tab.frac_correct_pool, bins, labels=labels), "early": early, "stop": stop,
                          "correct": corr, "fixed32": eval_tab["correct@32"]})
        by = t.groupby("bin", observed=False).agg(cases=("early", "size"), early_stop_pct=("early", "mean"), mean_stop=("stop", "mean"),
                                                 accuracy=("correct", "mean"), fixed32_accuracy=("fixed32", "mean")).reset_index()
        by["early_stop_pct"] *= 100
        out[fam] = {"order-question cases": int(len(stop)), "early stops": int(early.sum()),
                    "incorrect early stops": int(wrong_early.sum()),
                    "incorrect early stops that fixed-32 would have answered correctly": int(rec.sum()),
                    "by_fraction_correct_in_pool": by.to_dict(orient="records")}
    return out


# ============================================================================ examples
def examples(df, pools, orders, sp, vlm, chosen, name, row0, n=3):
    """Stated rule: the first n held-out questions (split order) showing each behaviour, under order 0, N = 32."""
    meth = MAIN_QUALITY_METHOD[name]
    ex = {"selection_disagreement": [], "premature_stop": []}
    for q in sp["eval"]:
        p, perm = pools[q], orders[q][0]
        sel = select_all(p, perm, vlm)
        if len(ex["selection_disagreement"]) < n and p.g[sel["SC"]] != p.g[sel[meth]]:
            votes = pd.Series([p.group_names[g] for g in p.g]).value_counts().head(5).to_dict()
            ex["selection_disagreement"].append({
                "sample_id": q, "gold": str(df.gold_answer.iloc[row0[q]]), "votes_top5": votes,
                "SC": {"rollout": sel["SC"], "answer": p.group_names[p.g[sel["SC"]]], "correct": bool(p.correct[sel["SC"]])},
                meth: {"rollout": sel[meth], "answer": p.group_names[p.g[sel[meth]]], "correct": bool(p.correct[sel[meth]])},
                "fraction_correct_in_pool": float(p.correct.mean())})
        ch = chosen.get("quality")
        if ch and ch["tau"] is not None and len(ex["premature_stop"]) < n:
            for k in CHECKPOINTS:
                st = stop_stats(p, perm[:k], vlm)
                if st["unique"] and not st["leader_no_answer"] and st["share"] * k >= ch["tau"] * k - 1e-9 and st["agree"] \
                        and st["margin"] >= (ch["gamma"] or 0) - 1e-12:
                    if not p.correct[st["sc_sel"]]:
                        full = stop_stats(p, perm, vlm)
                        ex["premature_stop"].append({
                            "sample_id": q, "gold": str(df.gold_answer.iloc[row0[q]]), "stopped_at": k,
                            "answer_at_stop": p.group_names[p.g[st["sc_sel"]]], "share_at_stop": st["share"], "rwc_margin": st["margin"],
                            "answers_seen": [p.group_names[g] for g in p.g[perm[:k]]],
                            "fixed32_answer": p.group_names[p.g[full["sc_sel"]]], "fixed32_correct": bool(p.correct[full["sc_sel"]]),
                            "fraction_correct_in_pool": float(p.correct.mean())})
                    break
        if len(ex["selection_disagreement"]) >= n and len(ex["premature_stop"]) >= n:
            break
    return ex


# ============================================================================ analyze
def analyze(out, datasets=("math", "mathvista"), checks=None, log=print):
    C = checks or Checks()
    out = Path(out)
    adir = out / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        if not (out / "prep" / f"{name}_rollouts.parquet").exists():
            print(f"(skipping {name}: not prepared)")
            continue
        vlm = DATASETS[name]
        t0 = time.time()
        df, pools, sp, orders, row0 = load_prepared(out, name)
        log(f"==== {name}: {len(pools)} questions")
        sel, cov = run_selection(pools, orders, vlm, log)
        sel.to_parquet(adir / f"{name}_selection_rows.parquet", index=False)
        cov.to_parquet(adir / f"{name}_coverage_rows.parquet", index=False)
        n1 = sel[sel.N == 1].pivot_table(index=["sample_id", "order"], columns="method", values="selected_rollout")
        C.check(bool((n1.nunique(axis=1) == 1).all()), f"{name}: at N = 1 every rule returns the only candidate", True,
                bool((n1.nunique(axis=1) == 1).all()))
        mc = sel.merge(cov, on=["sample_id", "order", "N"])
        C.check(bool((mc.coverage | ~mc.correct).all()), f"{name}: no rule is ever correct without a correct candidate (accuracy <= coverage)",
                True, bool((mc.coverage | ~mc.correct).all()))
        cv = cov.sort_values(["sample_id", "order", "N"]).groupby(["sample_id", "order"]).coverage.apply(lambda s: bool(np.all(np.diff(s.astype(int)) >= 0)))
        C.check(bool(cv.all()), f"{name}: oracle coverage never decreases along a nested order", True, bool(cv.all()))
        perq, summ, groups = summarise_selection(sel, cov, sp["eval"])
        perq.to_csv(adir / f"{name}_per_question.csv", index=False)
        summ.to_csv(adir / f"{name}_summary.csv", index=False)
        groups.to_csv(adir / f"{name}_answer_groups.csv", index=False)
        boot = bootstrap_table(perq, methods_for(vlm), D.BUDGETS)
        boot.to_csv(adir / f"{name}_bootstrap.csv", index=False)
        piv = summ.pivot(index="N", columns="method", values="accuracy")[methods_for(vlm)]
        log((100 * piv).round(1).to_string())
        cov32 = summ[summ.method == "FirstAns"].set_index("N").oracle_coverage
        C.observe(f"{name}: held-out oracle coverage by N (%)", {int(k): round(100 * v, 1) for k, v in cov32.items()})
        # Step 3
        pq3, s3, cvt, cvs = signal_analysis(pools, sp["eval"], vlm)
        pq3.to_csv(adir / f"{name}_signals_per_question.csv", index=False)
        s3.to_csv(adir / f"{name}_signals.csv", index=False)
        log(s3.round(3).to_string(index=False))
        if vlm:
            cvt.to_csv(adir / f"{name}_conf_vision_per_rollout.csv", index=False)
            write_json(adir / f"{name}_conf_vision_summary.json", cvs)
            C.observe(f"{name}: confidence-vision correlation within rollouts", cvs)
            write_json(adir / f"{name}_trajectories.json", illustrative_trajectories(pools, sp["eval"]))
        # Step 5
        dev_tab = stopping_table(pools, orders, sp["dev"], vlm)
        eval_tab = stopping_table(pools, orders, sp["eval"], vlm)
        grid, chosen = choose_settings(dev_tab)
        grid.to_csv(adir / f"{name}_adaptive_dev_grid.csv", index=False)
        write_json(adir / f"{name}_adaptive_chosen.json", {"quality_score": QUALITY_SCORE[name], **chosen})
        C.observe(f"{name}: stopping settings chosen on development (frozen before held-out)", chosen)
        ad, _ = adaptive_heldout(eval_tab, chosen, summ)
        ad.to_csv(adir / f"{name}_adaptive_heldout.csv", index=False)
        log(ad.round(3).to_string(index=False))
        write_json(adir / f"{name}_early_stops.json", early_stop_analysis(eval_tab, chosen))
        eval_tab.to_parquet(adir / f"{name}_stopping_rows.parquet", index=False)
        write_json(adir / f"{name}_examples.json", examples(df, pools, orders, sp, vlm, chosen, name, row0))
        C.observe(f"{name}: analysis time", f"{time.time() - t0:.0f} s")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "analyze", "all"])
    ap.add_argument("--data_root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/t3")
    ap.add_argument("--mathvista_meta", default=None, help="saved mathvista_meta.json (e.g. from an earlier run's output)")
    ap.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    a = ap.parse_args()
    C = Checks()
    if a.stage in ("prep", "all"):
        prep(a.data_root, a.out, a.mathvista_meta, a.datasets, C)
    if a.stage in ("analyze", "all"):
        analyze(a.out, a.datasets, C)
    C.summary()
    C.save(Path(a.out) / f"t3_offline_{a.stage}_{'_'.join(a.datasets)}_checks.json")


if __name__ == "__main__":
    main()
