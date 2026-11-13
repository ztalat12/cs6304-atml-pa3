"""Task 3: extra failure analyses computed from the saved results only (no model, no supplied data needed).

    python -m task3.t3_extra --results results/task3/t3 --out results/task3/t3_extra

1. sc_misses           when self-consistency misses although a correct candidate was available: did the NO_ANSWER group
                       win the vote, or a wrong answer? (held-out, % of question-orders, every N)
2. vote_without_no_answer
                       DIAGNOSTIC ONLY (not one of the compared methods; the manual keeps NO_ANSWER in voting):
                       self-consistency if the NO_ANSWER group could not win, over the same saved orders
3. unreadable_answers  MathVista answers the official normalisation rules cannot read (ranges, LaTeX fractions, "\\%",
                       units): how many math-verify would judge equal to the gold answer (a sensitivity check of the scorer)
4. live_revisions      which revisions changed the answer, and whether the initial response had been cut off
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import write_json  # noqa: E402


def sc_misses(res, name, ev):
    sel = pd.read_parquet(res / "analysis" / f"{name}_selection_rows.parquet")
    cov = pd.read_parquet(res / "analysis" / f"{name}_coverage_rows.parquet")
    e = sel[sel.sample_id.isin(ev) & (sel.method == "SC")].merge(cov, on=["sample_id", "order", "N"])
    n = len(ev) * e.order.nunique()
    rows = []
    for N, g in e.groupby("N"):
        miss = g[g.coverage & ~g.correct]
        rows.append({"dataset": name, "N": int(N), "SC accuracy (%)": 100 * g.correct.sum() / n,
                     "missed, NO_ANSWER won (%)": 100 * (miss.selected_group == "NO_ANSWER").sum() / n,
                     "missed, a wrong answer won (%)": 100 * (miss.selected_group != "NO_ANSWER").sum() / n,
                     "no correct candidate (%)": 100 * (~g.coverage).sum() / n})
    return rows


def vote_without_no_answer(res, name, ev, budgets=(1, 2, 4, 8, 16, 32)):
    df = pd.read_parquet(res / "prep" / f"{name}_rollouts.parquet")
    o = np.load(res / "prep" / f"{name}_orders.npz", allow_pickle=True)
    orders = dict(zip(o["ids"].tolist(), o["orders"]))
    g = {q: (s.group.to_numpy(), s.correct.to_numpy()) for q, s in df.groupby("sample_id")}
    out = {"dataset": name}
    for N in budgets:
        acc = []
        for q in ev:
            grp, cor = g[q]
            a = []
            for perm in orders[q]:
                real = [i for i in perm[:N] if grp[i] != "NO_ANSWER"]
                if not real:
                    a.append(False)
                    continue
                cnt = {}
                for pos, i in enumerate(real):
                    cnt.setdefault(grp[i], [0, pos, i])[0] += 1
                a.append(bool(cor[min(cnt.values(), key=lambda v: (-v[0], v[1]))[2]]))
            acc.append(np.mean(a))
        out[f"N={N} (%)"] = 100 * float(np.mean(acc))
    pq = df[df.sample_id.isin(ev)].groupby("sample_id").no_answer.mean()
    out["held-out questions with >= half NO_ANSWER rollouts"] = int((pq >= 0.5).sum())
    return out


def unreadable_answers(res):
    from task3.t3_data import _mv
    parse, verify = _mv()
    df = pd.read_parquet(res / "prep" / "mathvista_rollouts.parquet")
    meta = json.load(open(res / "prep" / "mathvista_meta.json"))["meta"]
    ext = df.extracted_answer
    fail = df[df.no_answer & ext.notna() & (ext.astype(str).str.strip() != "")].copy()
    cache = {}

    def eq(a, gld):
        if (a, gld) not in cache:
            try:
                cache[(a, gld)] = bool(verify(parse(str(gld)), parse(str(a))))
            except Exception:  # noqa: BLE001
                cache[(a, gld)] = False
        return cache[(a, gld)]

    fail["answer_type"] = fail.sample_id.map(lambda q: meta[q]["answer_type"])
    fail["math_verify_equal_to_gold"] = [eq(a, gd) for a, gd in zip(fail.extracted_answer, fail.gold_answer)]
    null = df[ext.isna() | (ext.astype(str).str.strip() == "")]
    null_types = null.sample_id.map(lambda q: meta[q]["question_type"]).value_counts().to_dict()
    return {"rollouts": int(len(df)), "null or empty extraction": int(len(null)),
            "null extraction by question type": null_types, "null extraction among capped rollouts": int(null.hit_max_new_tokens.sum()),
            "extracted but unreadable by the official rules": int(len(fail)),
            "unreadable, by answer type": fail.answer_type.value_counts().to_dict(),
            "unreadable but math-verify-equal to gold": int(fail.math_verify_equal_to_gold.sum()),
            "questions with such a rollout": int(fail[fail.math_verify_equal_to_gold].sample_id.nunique()),
            "most common unreadable answers": fail.extracted_answer.astype(str).value_counts().head(12).to_dict(),
            "examples equal to gold": fail[fail.math_verify_equal_to_gold][["sample_id", "extracted_answer", "gold_answer"]]
            .drop_duplicates("extracted_answer").head(10).to_dict(orient="records")}


def live_revisions(res):
    p = pd.read_csv(res / "live" / "live_per_question.csv")
    ch = p[p.revision_changed_answer]
    comp = p[~p.initial_truncated]
    return {"questions": int(len(p)), "revision changed the answer": int(len(ch)),
            "of which the initial response had been cut off at 1,024 tokens": int(ch.initial_truncated.sum()),
            "initial responses cut off": int(p.initial_truncated.sum()),
            "completed initial responses whose answer the revision changed": int(comp.revision_changed_answer.sum()),
            "completed initial responses": int(len(comp)),
            "changed cases": ch[["sample_id", "gold", "initial_answer", "initial_correct", "revised_answer", "revised_correct",
                                 "initial_truncated"]].to_dict(orient="records"),
            "parallel: questions with >= 1 call cut off": int((p.parallel_truncated_calls > 0).sum()),
            "parallel: final vote NO_ANSWER": int(p.parallel_no_answer.sum()),
            "single right, parallel right": int((p.single_correct & p.parallel_correct).sum()),
            "single right, parallel wrong": int((p.single_correct & ~p.parallel_correct).sum()),
            "single wrong, parallel right": int((~p.single_correct & p.parallel_correct).sum()),
            "single wrong, parallel wrong": int((~p.single_correct & ~p.parallel_correct).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    miss, vote = [], []
    for name in ["math", "mathvista"]:
        if not (res / "analysis" / f"{name}_selection_rows.parquet").exists():
            continue
        ev = json.load(open(res / "prep" / f"{name}_splits.json"))["eval"]
        miss += sc_misses(res, name, ev)
        vote.append(vote_without_no_answer(res, name, ev))
    pd.DataFrame(miss).to_csv(out / "t3_sc_misses.csv", index=False)
    pd.DataFrame(vote).to_csv(out / "t3_vote_without_no_answer.csv", index=False)
    print(pd.DataFrame(miss).round(1).to_string(index=False))
    print(pd.DataFrame(vote).round(1).to_string(index=False))
    if (res / "prep" / "mathvista_rollouts.parquet").exists():
        u = unreadable_answers(res)
        write_json(out / "t3_mathvista_unreadable_answers.json", u)
        print(json.dumps({k: v for k, v in u.items() if k != "examples equal to gold"}, indent=1, ensure_ascii=False))
    if (res / "live" / "live_per_question.csv").exists():
        lv = live_revisions(res)
        write_json(out / "t3_live_revisions.json", lv)
        print(json.dumps({k: v for k, v in lv.items() if k != "changed cases"}, indent=1, default=str))


if __name__ == "__main__":
    main()
