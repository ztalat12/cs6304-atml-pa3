"""Offline tests for Task 3 (CPU, synthetic data, tiny random model). Run:  python tests/test_task3.py /path/to/tinyqwen

  1  answer extraction (balanced boxes), MATH grouping/grading with math-verify, MathVista official normalisation
  2  selection rules: ties, RWC ranks and margin, VisConf calibration on the visible prefix only
  3  splits and random orders are reproducible; stopping simulation on a hand-made table
  4  the whole offline pipeline (prep + analyze + report) on synthetic data shaped like the supplied files
  5  live generation bookkeeping with a tiny model: five calls per question, seeds, costs, resume
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TINY = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinyqwen"
W = Path("/tmp/t3_test")
shutil.rmtree(W, ignore_errors=True)
from make_synthetic_t3 import make  # noqa: E402

root, totals = make(W, 60)
os.environ.update({"PA3_T3_NQ": "60", "PA3_T3_NDEV": "12", "PA3_T3_NLIVE": "5", "PA3_T3_README_TOTALS": json.dumps(totals),
                   "PA3_T3_MODEL": TINY, "PA3_T3_REVISION": ""})

from common.utils import Checks, read_jsonl  # noqa: E402
from task3 import t3_data as D  # noqa: E402
from task3 import t3_methods as M  # noqa: E402
from task3 import t3_offline as O  # noqa: E402

C = Checks()

print("\n### 1. answers")
cases = {"so \\boxed{\\frac{1}{2}}": "\\frac{1}{2}", "\\boxed{1} then \\boxed{ x^{2}+1 }": "x^{2}+1", "\\boxed{3": None,
         "\\boxed{}": None, "no box": None, "\\boxed{\\text{(A)}}": "\\text{(A)}"}
got = {k: D.last_boxed(k) for k in cases}
C.check(got == cases, "last \\boxed{...} with balanced braces; absent / empty / unclosed -> None", cases, got)
keys, corr = D.math_groups_and_grades(["0.5", "\\frac{1}{2}", None, "3", "", "1/2"], "\\frac12")
C.check(keys == ["0.5", "0.5", D.NO_ANSWER, "3", D.NO_ANSWER, "0.5"] and corr == [True, True, False, False, False, True],
        "MATH: equivalent forms share a group (formed without gold); missing -> NO_ANSWER; graded by math-verify",
        "groups [0.5, 0.5, NO, 3, NO, 0.5], correct [T, T, F, F, F, T]", (keys, corr))
k2, c2 = D.math_groups_and_grades(["7"], "The answer is $\\boxed{7}$.")
C.check(c2 == [True], "MATH: a gold solution with a box is graded against its last boxed answer", [True], c2)
mc = {"question_type": "multi_choice", "answer_type": "text", "precision": None, "choices": ["red", "green", "blue"], "answer": "blue"}
norm = [D.mathvista_normalize(x, mc)[0] for x in ["C", "(C) blue", "blue", "bleu", "(b)"]]
C.check(norm == ["C", "C", "C", "C", "B"] and D.mathvista_gold(mc) == "C",
        "MathVista multiple choice -> option letters (letter, '(X) text', option text, most-similar fallback)", ["C", "C", "C", "C", "B"], norm)
it = {"question_type": "free_form", "answer_type": "integer", "precision": None, "choices": None, "answer": "7"}
fl = {"question_type": "free_form", "answer_type": "float", "precision": 2.0, "choices": None, "answer": "1.25"}
vals = [D.mathvista_normalize("7.0", it)[0], D.mathvista_normalize("seven", it)[0], D.mathvista_normalize("1.2500", fl)[0]]
C.check(vals == ["7", None, "1.25"], "MathVista integer / float (precision) rules; unparsable -> None (NO_ANSWER)", ["7", None, "1.25"], vals)
k3, c3, st = D.mathvista_groups_and_grades(["7", "seven", None, "8"], it)
C.check(k3 == ["7", D.NO_ANSWER, D.NO_ANSWER, "8"] and c3 == [True, False, False, False], "MathVista groups and grades",
        "[7, NO, NO, 8]", k3)

print("\n### 2. selection rules")
g = np.array([0, 1, 1, 0, 2])
C.check(M.self_consistency(np.array([0, 1, 2, 3]), g)[0] == 0, "self-consistency tie -> earliest supporting rollout in the order", 0,
        M.self_consistency(np.array([0, 1, 2, 3]), g)[0])
C.check(M.self_consistency(np.array([1, 0, 3, 2]), g)[0] == 1, "...in the CURRENT order (order 1, 0, 3, 2 -> rollout 1)", 1,
        M.self_consistency(np.array([1, 0, 3, 2]), g)[0])
# RWC example from the explanation: B (rank 1), A (rank 2), A (rank 3) -> B; majority -> A
gB = np.array([1, 0, 0])
sel, margin = M.rwc(np.array([0, 1, 2]), np.array([0.9, 0.5, 0.1]), gB)
C.check(sel == 0 and abs(margin - (1.0 - 0.5) / 1.5) < 1e-12 and M.self_consistency(np.array([0, 1, 2]), gB)[0] == 1,
        "RWC: w = (N - r)/(N - 1); B=1 beats A=0.5+0, margin (1-0.5)/1.5; majority picks A", (0, 1 / 3), (sel, round(margin, 4)))
sel2, _ = M.rwc(np.array([0, 1, 2, 3]), np.array([0.5, 0.5, 0.2, 0.9]), np.array([0, 1, 2, 3]))
C.check(sel2 == 3, "RWC with tied scores uses average ranks", 3, sel2)
C.check(M.highest(np.array([4, 2, 7]), np.array([np.nan, 0.3, 0.3])) == 2, "highest score: NaN ranks last, ties -> earliest", 2,
        M.highest(np.array([4, 2, 7]), np.array([np.nan, 0.3, 0.3])))
kl = np.array([1, 2, 3, 10, 20, 30, 4, 5], np.float32)
v = np.array([.1, .2, .3, .4, .5, .6, .7, .8], np.float32)
pool = M.QuestionPool([3, 3, 2], [0, 0, 0], [2, 20, 4.5], ["a", "b", "a"], [1, 0, 1], [0, 0, 0],
                      V=np.array([.2, .5, .75]), kl=kl, v=v, offsets=np.array([0, 3, 6, 8]))
Z, Zt, Cc, (q05, q95) = pool.visconf(np.array([0, 2]))
pooled = np.array([1, 2, 3, 4, 5.0])
lo, hi = np.percentile(pooled, [5, 95])
c = np.clip((pooled - lo) / (hi - lo), 0, 1)
expC = np.array([c[:3].mean(), c[3:].mean()])
expZt = np.array([(c[:3] * [.1, .2, .3]).mean(), (c[3:] * [.7, .8]).mean()])
C.check(np.allclose(Cc, expC) and np.allclose(Z, [.2, .75] * expC) and np.allclose(Zt, expZt),
        "VisConf: 5th/95th percentiles of the VISIBLE rollouts' token KLs, clip, average; Z = V*C; tokenwise mean(c*v)",
        np.round(expC, 4).tolist(), np.round(Cc, 4).tolist())
pool.kl[3:6] = 1e6    # an unseen rollout must not affect the calibration
C.check(np.allclose(pool.visconf(np.array([0, 2]))[2], expC), "VisConf ignores rollouts that are not yet revealed")
C.check(np.allclose(Zt - Z, [np.cov(c[:3], [.1, .2, .3], bias=True)[0, 1], np.cov(c[3:], [.7, .8], bias=True)[0, 1]]),
        "tokenwise Z - Z equals the within-rollout covariance of c and v")

print("\n### 3. splits, orders, stopping")
ids = [f"test:{i}" for i in range(60)]
s1, s2 = D.make_splits(ids), D.make_splits(list(reversed(ids)))
C.check(s1 == s2 and s1["dev"] == np.array(sorted(ids), dtype=object)[np.random.default_rng(6304).permutation(60)][:12].tolist(),
        "split: sort IDs lexicographically, permute with default_rng(6304); input order irrelevant")
o1, o2 = D.make_orders(sorted(ids)), D.make_orders(sorted(ids))
C.check(all((o1[q] == o2[q]).all() and o1[q].shape == (10, 32) and all(sorted(r) == list(range(32)) for r in o1[q]) for q in ids),
        "orders: 10 seeded permutations of the 32 rollout IDs per question, reproducible")
tab = pd.DataFrame({"sample_id": ["a", "b"], "order": [0, 0], "correct@32": [True, False], "cost@32": [320, 320],
                    **{f"{k}@{n}": v for n in O.CHECKPOINTS for k, v in [("unique", [True, True]), ("leader_no_answer", [False, n == 4]),
                                                                       ("share", [0.75, 1.0]), ("agree", [True, True]),
                                                                       ("margin", [0.3, 0.6]), ("correct", [True, False]),
                                                                       ("cost", [10 * n, 10 * n])]}})
stop, corr, cost = O.simulate(tab, "agreement", 0.75, None)
C.check(list(stop) == [4, 8] and list(cost) == [40, 80], "agreement stop: unique, non-NO_ANSWER leader with share >= tau",
        "[4, 8]", list(stop))
stop, _, _ = O.simulate(tab, "quality", 0.75, 0.5)
C.check(list(stop) == [32, 8], "quality-aware stop also needs RWC margin >= gamma", "[32, 8]", list(stop))
stop, _, _ = O.simulate(tab, "agreement", None, None)
C.check(list(stop) == [32, 32], "fallback policy always consumes 32", "[32, 32]", list(stop))

print("\n### 4. offline pipeline on synthetic data")
out = W / "out"
r = subprocess.run([sys.executable, "-m", "task3.t3_offline", "all", "--data_root", str(root), "--out", str(out),
                    "--mathvista_meta", str(W / "mathvista_meta.json")], cwd=ROOT, capture_output=True, text=True, env=os.environ)
C.check(r.returncode == 0 and "0 FAIL" in r.stdout, "prep + analyze run and their own checks pass", "0 FAIL",
        [l for l in r.stdout.splitlines() if l.startswith("====")][-1:] or r.stderr[-1500:])
a = out / "analysis"
need = [f"{n}_{f}" for n in ["math", "mathvista"] for f in ["summary.csv", "bootstrap.csv", "signals.csv", "adaptive_dev_grid.csv",
                                                           "adaptive_heldout.csv", "examples.json", "early_stops.json"]] + \
    ["mathvista_conf_vision_summary.json", "mathvista_trajectories.json"]
C.check(all((a / f).exists() for f in need), "analysis wrote every table", need, [f for f in need if not (a / f).exists()])
if (a / "math_summary.csv").exists():
    s = pd.read_csv(a / "math_summary.csv")
    tot = s.accuracy + s.missed_with_correct_available + s.no_correct_candidate
    C.check(bool(np.allclose(tot, 1)), "decomposition: correct + missed + no correct candidate = 100%", 1, float(tot.max()))
    x = s[s.N == 32]
    C.check(bool((x.accuracy <= x.oracle_coverage + 1e-12).all()), "every rule's accuracy <= oracle coverage")
    sel = pd.read_parquet(a / "math_selection_rows.parquet")
    f1 = sel[(sel.method == "FirstAns")]
    C.check(bool(f1.groupby(["sample_id", "order"]).selected_rollout.nunique().eq(1).all()), "FirstAns keeps its first candidate at every N")
    ch = json.load(open(a / "math_adaptive_chosen.json"))
    grid = pd.read_csv(a / "math_adaptive_dev_grid.csv")
    q = grid[(grid.family == "quality") & grid.eligible]
    C.check(abs(q.mean_cost.min() - ch["quality"]["dev_mean_cost"]) < 1e-9, "quality-aware setting = cheapest eligible on development",
            q.mean_cost.min(), ch["quality"]["dev_mean_cost"])
rep = W / "rep"
r2 = subprocess.run([sys.executable, "-m", "task3.t3_report", "--results", str(out), "--out", str(rep)], cwd=ROOT, capture_output=True,
                    text=True, env=os.environ)
figs = ["t3_accuracy_curves.png", "t3_failure_decomposition.png", "t3_mathvista_signals.png", "t3_math_methods.csv"]
C.check(r2.returncode == 0 and all((rep / f).exists() for f in figs), "report tables and figures build from saved results only", figs,
        r2.stderr[-800:] if r2.returncode else [f for f in figs if not (rep / f).exists()])

print("\n### 5. live generation bookkeeping (tiny model, 2 questions)")
lo_ = W / "live"
r3 = subprocess.run([sys.executable, "-m", "task3.t3_live", "all", "--data_root", str(root), "--out", str(lo_), "--limit", "2"],
                    cwd=ROOT, capture_output=True, text=True, env=os.environ)
calls = read_jsonl(lo_ / "live" / "live_calls.jsonl") if (lo_ / "live" / "live_calls.jsonl").exists() else []
C.check(r3.returncode == 0 and len(calls) == 10, "five calls per question saved", 10, len(calls) if r3.returncode == 0 else r3.stderr[-1500:])
if calls:
    seeds = {(c["q_index"], c["call"]): c["seed"] for c in calls}
    C.check(seeds[(1, "parallel_2")] == 6304 + 10 + 2 and seeds[(0, "revision")] == 6304 + 4, "seed = 6304 + 10 x question + call index")
    rv = [c for c in calls if c["call"] == "revision"][0]
    p1 = [c for c in calls if c["call"] == "parallel_1" and c["q_index"] == rv["q_index"]][0]
    C.check(p1["response"] in rv["prompt"] and "Review the solution above" in rv["prompt"] and rv["input_tokens"] > p1["input_tokens"],
            "revision prompt = original prompt + first parallel response + review instruction (longer prompt counted)")
    C.check(all(c["cap"] == {"single": 3072, "revision": 2048}.get(c["call"], 1024) for c in calls) and
            all(c["output_tokens"] <= c["cap"] for c in calls), "token caps 3,072 / 3 x 1,024 / 2,048")
    pq_ = pd.read_csv(lo_ / "live" / "live_per_question.csv")
    one = pq_.iloc[0]
    mine = [c for c in calls if c["sample_id"] == one.sample_id]
    ref_out = sum(c["output_tokens"] for c in mine if c["call"] in ("parallel_1", "revision"))
    C.check(int(one.refinement_output_tokens) == ref_out, "refinement cost charges the shared initial response + the revision",
            ref_out, int(one.refinement_output_tokens))
    # resume: drop the last two calls, run again with the earlier file attached -> only those two are generated
    keep = calls[:-2]
    att = W / "attached_calls.jsonl"
    with open(att, "w") as f:
        for c in keep:
            f.write(json.dumps(c) + "\n")
    lo2 = W / "live2"
    r4 = subprocess.run([sys.executable, "-m", "task3.t3_live", "generate", "--data_root", str(root), "--out", str(lo2), "--limit", "2",
                         "--resume_from", str(att)], cwd=ROOT, capture_output=True, text=True, env=os.environ)
    c2 = read_jsonl(lo2 / "live" / "live_calls.jsonl") if r4.returncode == 0 else []
    same = [a_["response"] for a_ in c2[:8]] == [b_["response"] for b_ in keep]
    C.check(r4.returncode == 0 and len(c2) == 10 and same, "resume: earlier calls carried over unchanged, only missing calls generated",
            10, len(c2) if r4.returncode == 0 else r4.stderr[-800:])
    regen = {(c["q_index"], c["call"]): c["response"] for c in c2[8:]}
    orig = {(c["q_index"], c["call"]): c["response"] for c in calls[-2:]}
    C.check(regen == orig, "a regenerated call reproduces the original response (same seed)", True, regen == orig)

print("\n### 6. extra failure analyses from saved results only")
if (out / "analysis").exists() and (lo_ / "live").exists():
    shutil.copytree(lo_ / "live", out / "live", dirs_exist_ok=True)
    r5 = subprocess.run([sys.executable, "-m", "task3.t3_extra", "--results", str(out), "--out", str(W / "extra")], cwd=ROOT,
                        capture_output=True, text=True, env=os.environ)
    ok = r5.returncode == 0 and all((W / "extra" / f).exists() for f in
                                    ["t3_sc_misses.csv", "t3_vote_without_no_answer.csv", "t3_mathvista_unreadable_answers.json",
                                     "t3_live_revisions.json"])
    C.check(ok, "t3_extra runs on the saved results", 0, r5.returncode if ok else r5.stderr[-1000:])
    if ok:
        m = pd.read_csv(W / "extra" / "t3_sc_misses.csv")
        s_ = pd.read_csv(out / "analysis" / "math_summary.csv")
        sc = s_[s_.method == "SC"].set_index("N")
        mm = m[m.dataset == "math"].set_index("N")
        tot = mm["SC accuracy (%)"] + mm["missed, NO_ANSWER won (%)"] + mm["missed, a wrong answer won (%)"] + mm["no correct candidate (%)"]
        C.check(bool(np.allclose(tot, 100)) and bool(np.allclose(mm["SC accuracy (%)"], 100 * sc.loc[mm.index, "accuracy"])),
                "SC miss breakdown adds up to 100% and matches the main summary", 100, tot.round(6).tolist())
        v = pd.read_csv(W / "extra" / "t3_vote_without_no_answer.csv")
        n1 = v[v.dataset == "math"]["N=1 (%)"].iloc[0]
        C.check(abs(n1 - 100 * sc.loc[1, "accuracy"]) < 1e-9, "vote-without-NO_ANSWER diagnostic equals self-consistency at N = 1",
                round(100 * sc.loc[1, "accuracy"], 6), round(n1, 6))

sys.exit(C.summary())
