"""Task 3 data: load and join the supplied rollouts, check alignment, make splits and orders, group and grade answers.

Supplied data (README.MD of ATML-PA3-Data): one folder per model, 32,000 rows per file, joined on
(dataset, sample_id, rollout_id); every per-token list aligns with generated_token_ids (stop tokens excluded).

Answers (grouped WITHOUT gold answers; graded with the same rules):
  MATH       math-verify (pinned) on the supplied extracted answer, wrapped as \\boxed{...} so LaTeX is parsed.
             Two answers share a group when math-verify finds them equivalent (either direction); each new answer is
             compared with the first answer of every existing group, in rollout-ID order (decision D-T3-2).
  MathVista  MathVista's official normalisation (evaluation/calculate_score.py: option letter / most-similar option,
             int(float()) for integers, round(float(), precision) for floats, str() for lists), using the released
             testmini metadata; multiple-choice answers are represented as option letters. Groups = equal normalised
             strings. The gold answer goes through the same normalisation.
  Missing    A null / empty extraction, or a MathVista answer the official rules cannot normalise (they return None),
             goes to one NO_ANSWER group, which is always graded incorrect and stays in voting and cost accounting.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import SEED  # noqa: E402

KEYS = ["dataset", "sample_id", "rollout_id"]
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
N_ROLLOUTS = 32
# the PA3_T3_* environment overrides exist ONLY for the offline unit tests on small synthetic data
N_QUESTIONS = int(os.environ.get("PA3_T3_NQ", 1000))
N_DEV = int(os.environ.get("PA3_T3_NDEV", 200))
N_LIVE = int(os.environ.get("PA3_T3_NLIVE", 50))
N_ORDERS = 10
BUDGETS = (1, 2, 4, 8, 16, 32)
NO_ANSWER = "NO_ANSWER"
# totals stated in the supplied README.MD (checked after loading)
README_TOTALS = {"math": {"retained_tokens": 20_742_094, "capped": 523, "null_extracted": 521},
                 "mathvista": {"retained_tokens": 8_961_703, "capped": 487, "null_extracted": 3_032}}
if os.environ.get("PA3_T3_README_TOTALS"):
    README_TOTALS = json.loads(os.environ["PA3_T3_README_TOTALS"])
MATHVISTA_REPO = "AI4Math/MathVista"


# ----------------------------------------------------------------------------- loading
def find_folders(roots):
    """{'math': path, 'mathvista': path} for the supplied model folders found under any of `roots`."""
    out = {}
    for root in roots:
        for name, folder in FOLDERS.items():
            if name in out:
                continue
            hits = sorted(Path(root).rglob(f"{folder}/generations.parquet"))
            if hits:
                out[name] = hits[0].parent
    return out


def _flat_list_column(table, name):
    """A list<float/int> column as (flat numpy values, offsets) without building Python lists."""
    import pyarrow as pa
    col = table.column(name).combine_chunks()
    if isinstance(col, pa.ChunkedArray):
        col = col.chunk(0) if col.num_chunks else pa.array([], type=col.type)
    offsets = np.asarray(col.offsets)
    values = np.asarray(col.values)
    if len(offsets) and offsets[0] != 0:
        values, offsets = values[offsets[0]:], offsets - offsets[0]
    return values, offsets


def _segment_means(values, offsets):
    """Mean of each segment; NaN for an empty segment."""
    lens = np.diff(offsets)
    sums = np.add.reduceat(values.astype(np.float64), np.minimum(offsets[:-1], max(len(values) - 1, 0))) if len(values) else np.zeros(len(lens))
    sums = np.where(lens > 0, sums, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(lens > 0, sums / np.maximum(lens, 1), np.nan)


def load_folder(folder, vlm, checks=None, name=""):
    """Join the parquet files of one model folder. Returns (rollouts DataFrame sorted by (sample_id, rollout_id),
    token dict). The DataFrame holds per-rollout scalars; the token dict holds flat per-token arrays + offsets
    (only what the analysis needs: kl_u_p and, for the VLM, the three attention totals)."""
    import pyarrow.parquet as pq
    folder = Path(folder)
    gen = pq.read_table(folder / "generations.parquet")
    prob = pq.read_table(folder / "probability_metrics.parquet")
    att = pq.read_table(folder / "attention_metrics.parquet") if vlm else None

    def keyframe(t):
        return t.select(KEYS).to_pandas()

    kg, kp = keyframe(gen), keyframe(prob)
    order_g = kg.sort_values(["sample_id", "rollout_id"], kind="mergesort").index.to_numpy()
    kg_sorted = kg.iloc[order_g].reset_index(drop=True)

    def aligned_take(t, kt):
        idx = kt.reset_index().merge(kg_sorted.reset_index(), on=KEYS, how="right", suffixes=("_t", "_g"))
        ok = bool(idx["index_t"].notna().all()) and len(idx) == len(kg_sorted)
        if not ok:
            raise ValueError(f"{name}: {int(idx['index_t'].isna().sum())} rollouts of generations.parquet have no match in "
                             f"another file on {KEYS}")
        return t.take(idx["index_t"].astype(np.int64).to_numpy()), ok

    prob_s, ok_p = aligned_take(prob, kp)
    gen_s = gen.take(order_g)
    if vlm:
        att_s, ok_a = aligned_take(att, keyframe(att))
    df = gen_s.select([c for c in gen_s.column_names if c not in ("generated_token_ids", "generated_text")]).to_pandas()
    df["generated_text"] = gen_s.column("generated_text").to_pylist()
    tok_ids, tok_off = _flat_list_column(gen_s, "generated_token_ids")
    lp, lp_off = _flat_list_column(prob_s, "logp")
    kl, kl_off = _flat_list_column(prob_s, "kl_u_p")
    n_tok = np.diff(tok_off)
    df["T"] = df["n_generated_text_tokens"].astype(int)
    df["L"] = _segment_means(lp, lp_off)
    df["K"] = _segment_means(kl, kl_off)
    tokens = {"kl": kl.astype(np.float32), "offsets": kl_off.astype(np.int64)}
    if checks is not None:
        nr = N_QUESTIONS * N_ROLLOUTS
        checks.check(len(df) == nr and len(prob) == nr and (att is None or len(att) == nr),
                     f"{name}: {nr:,} rollouts in every file", nr,
                     (len(df), len(prob)) + ((len(att),) if att is not None else ()))
        checks.check(bool(ok_p) and (not vlm or bool(ok_a)), f"{name}: every rollout joins across files on {KEYS}", True,
                     (bool(ok_p), bool(ok_a) if vlm else None))
        per_q = df.groupby("sample_id").rollout_id.agg(["count", "min", "max", "nunique"])
        checks.check(bool(((per_q["count"] == 32) & (per_q["nunique"] == 32) & (per_q["min"] == 0) & (per_q["max"] == 31)).all())
                     and len(per_q) == N_QUESTIONS, f"{name}: {N_QUESTIONS:,} questions x rollout IDs 0..31", f"{N_QUESTIONS} x 32",
                     f"{len(per_q)} questions, counts {sorted(per_q['count'].unique().tolist())}")
        same = (n_tok == df["T"].to_numpy()) & (np.diff(lp_off) == n_tok) & (np.diff(kl_off) == n_tok)
        checks.check(bool(same.all()), f"{name}: token lists aligned (token IDs = logp = kl_u_p = n_generated_text_tokens)",
                     f"all {len(df):,}", f"{int(same.sum()):,} aligned")
    if vlm:
        img, img_off = _flat_list_column(att_s, "img_attn_total")
        pr, pr_off = _flat_list_column(att_s, "prompt_text_attn_total")
        ge, ge_off = _flat_list_column(att_s, "generated_text_attn_total")
        cnt, cnt_off = _flat_list_column(att_s, "n_generated_text_tokens")
        denom = img.astype(np.float64) + pr + ge
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(denom > 0, img / denom, np.nan)
        tokens["v"] = v.astype(np.float32)
        df["V"] = _segment_means(np.nan_to_num(v, nan=0.0), kl_off)
        df["n_image_tokens"] = att_s.column("n_image_tokens").to_numpy()
        if checks is not None:
            same = all(np.array_equal(np.diff(o), n_tok) for o in (img_off, pr_off, ge_off, cnt_off))
            checks.check(same, f"{name}: attention lists aligned with the token lists", True, same)
            expect = np.arange(len(cnt)) - np.repeat(cnt_off[:-1], np.diff(cnt_off))
            checks.check(bool(np.array_equal(cnt.astype(np.int64), expect)),
                         f"{name}: attention n_generated_text_tokens lists are 0..T-1 (preceding-token counts)", True,
                         bool(np.array_equal(cnt.astype(np.int64), expect)))
            first = ge[cnt_off[:-1][np.diff(cnt_off) > 0]]
            checks.check(bool(np.allclose(first, 0)), f"{name}: first prediction has zero generated-text attention", 0,
                         float(np.abs(first).max()) if len(first) else 0)
            checks.observe(f"{name}: attention mass summed over the three groups (need not be 1)",
                           {"mean": round(float(np.nanmean(denom)), 4), "min": round(float(np.nanmin(denom)), 4),
                            "max": round(float(np.nanmax(denom)), 4)})
            checks.observe(f"{name}: tokens with zero total attention (v undefined, counted as 0 in Vᵢ)", int((denom <= 0).sum()))
    return df, tokens


def readme_totals_check(df, name, checks):
    t = README_TOTALS[name]
    ext = df["extracted_answer"]
    null = int((ext.isna() | (ext.astype(str).str.strip() == "")).sum())
    actual = {"retained_tokens": int(df["T"].sum()), "capped": int(df["hit_max_new_tokens"].sum()), "null_extracted": null}
    checks.check(actual == t, f"{name}: totals match the supplied README", t, actual)


# ----------------------------------------------------------------------------- splits and orders
def make_splits(ids, seed=SEED):
    """Sort the question IDs lexicographically, permute with default_rng(seed); first 200 = development."""
    s = np.array(sorted(ids), dtype=object)
    perm = s[np.random.default_rng(seed).permutation(len(s))]
    return {"sorted": s.tolist(), "permuted": perm.tolist(), "dev": perm[:N_DEV].tolist(), "eval": perm[N_DEV:].tolist(),
            "live": perm[N_DEV:N_DEV + N_LIVE].tolist()}


def make_orders(sorted_ids, seed=SEED, n_orders=N_ORDERS, n=N_ROLLOUTS):
    """Ten seeded permutations of the 32 rollout IDs per question: ONE generator default_rng(seed), questions visited
    in sorted-ID order, ten permutations each. Nested prefixes of each permutation give the budgets."""
    rng = np.random.default_rng(seed)
    return {q: np.stack([rng.permutation(n) for _ in range(n_orders)]) for q in sorted_ids}


# ----------------------------------------------------------------------------- MATH answers (math-verify)
def last_boxed(text):
    """The supplied export's rule (README.MD): find the LAST literal '\\boxed{', scan balanced braces (nested LaTeX kept),
    strip surrounding whitespace. Absent, empty or unclosed last box -> None; no numeric fallback."""
    if text is None:
        return None
    i = str(text).rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    s = str(text)
    k = j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                inner = s[j:k].strip()
                return inner or None
        k += 1
    return None


def _mv():
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
    return (lambda s: parse("\\boxed{" + s + "}", extraction_config=cfg)), verify


def _is_missing(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == ""


def math_groups_and_grades(answers, gold, parse=None, verify=None):
    """answers: 32 extracted strings (rollout-ID order). Returns (group key per rollout, correct per rollout).
    Groups are formed without the gold answer; grading compares each answer with the gold answer."""
    if parse is None:
        parse, verify = _mv()
    keys, correct = [None] * len(answers), [False] * len(answers)
    reps, parsed_cache, grade_cache = [], {}, {}
    gold = str(gold).strip()
    if "\\boxed{" in gold:                 # a full solution: grade against its last boxed answer
        gold = last_boxed(gold) or gold
    try:
        g = parse(gold)
    except Exception:  # noqa: BLE001
        g = []

    def P(s):
        if s not in parsed_cache:
            try:
                parsed_cache[s] = parse(s)
            except Exception:  # noqa: BLE001
                parsed_cache[s] = []
        return parsed_cache[s]

    def eq(a, b):
        if a == b:
            return True
        pa, pb = P(a), P(b)
        if not pa or not pb:
            return False
        try:
            return bool(verify(pa, pb) or verify(pb, pa))
        except Exception:  # noqa: BLE001
            return False

    for i, a in enumerate(answers):
        if _is_missing(a):
            keys[i] = NO_ANSWER
            continue
        s = str(a).strip()
        for r in reps:
            if eq(r, s):
                keys[i] = r
                break
        else:
            reps.append(s)
            keys[i] = s
        if s not in grade_cache:
            try:
                grade_cache[s] = bool(g) and bool(P(s)) and bool(verify(g, P(s)))
            except Exception:  # noqa: BLE001
                grade_cache[s] = False
        correct[i] = grade_cache[s]
    return keys, correct


# ----------------------------------------------------------------------------- MathVista answers (official rules)
def levenshtein(a, b):
    a, b = str(a), str(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mathvista_normalize(extraction, meta):
    """MathVista's normalize_extracted_answer, returning (normalised string or None, used_most_similar_fallback).
    Multiple-choice answers come back as option LETTERS (the official code returns the option text; the letter is
    its index, so equality is unchanged)."""
    qt, at, prec, choices = meta["question_type"], meta["answer_type"], meta.get("precision"), meta.get("choices") or []
    if qt == "multi_choice":
        x = extraction.strip() if isinstance(extraction, str) else str(extraction)
        letter = re.findall(r"\(([a-zA-Z])\)", x)
        if letter:
            x = letter[0].upper()
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        if x in letters:
            return x, False
        if not choices:
            return None, False
        d = [levenshtein(x, c) for c in choices]
        return letters[d.index(min(d))], True
    try:
        if at == "integer":
            return str(int(float(extraction))), False
        if at == "float":
            return str(round(float(extraction), int(prec))), False
        if at == "list":
            return str(extraction), False
    except Exception:  # noqa: BLE001
        return None, False
    return (str(extraction).strip() or None), False


def mathvista_gold(meta):
    """The gold answer under the same rules (multiple choice: the letter of the answer option)."""
    if meta["question_type"] == "multi_choice":
        choices = meta.get("choices") or []
        if meta["answer"] in choices:
            return chr(ord("A") + choices.index(meta["answer"]))
        return mathvista_normalize(meta["answer"], meta)[0]
    return mathvista_normalize(meta["answer"], meta)[0]


def mathvista_groups_and_grades(answers, meta):
    gold = mathvista_gold(meta)
    keys, correct, fallback, failed = [], [], 0, 0
    for a in answers:
        if _is_missing(a):
            keys.append(NO_ANSWER)
            correct.append(False)
            continue
        n, fb = mathvista_normalize(a, meta)
        fallback += fb
        if n is None:
            failed += 1
            keys.append(NO_ANSWER)
            correct.append(False)
        else:
            keys.append(n)
            correct.append(gold is not None and n == gold)
    return keys, correct, {"most_similar_fallback": fallback, "normalisation_failed": failed, "gold": gold}


def load_mathvista_meta(cache_dir, meta_json=None):
    """{pid: {question_type, answer_type, precision, choices, answer}} for testmini, plus the source revision.
    Uses `meta_json` (e.g. an earlier run's saved copy) when given; otherwise downloads only the testmini parquet."""
    if meta_json and Path(meta_json).exists():
        d = json.load(open(meta_json))
        return d["meta"], d["source"]
    from huggingface_hub import HfApi, snapshot_download
    import pyarrow.parquet as pq
    sha = HfApi().dataset_info(MATHVISTA_REPO).sha
    cols = ["pid", "question_type", "answer_type", "precision", "choices", "answer"]
    path = snapshot_download(MATHVISTA_REPO, repo_type="dataset", revision=sha, allow_patterns=["*testmini*.parquet"],
                             cache_dir=str(cache_dir))
    files = sorted(Path(path).rglob("*testmini*.parquet"))
    if files:
        t = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    else:                                   # different repository layout: fall back to the datasets loader (same revision)
        from datasets import load_dataset
        ds = load_dataset(MATHVISTA_REPO, split="testmini", revision=sha, cache_dir=str(cache_dir))
        t = ds.select_columns(cols).to_pandas()
        files = [Path("datasets.load_dataset(split='testmini')")]
    meta = {}
    for r in t.itertuples(index=False):
        ch = r.choices
        meta[str(r.pid)] = {"question_type": r.question_type, "answer_type": r.answer_type,
                            "precision": None if r.precision is None or (isinstance(r.precision, float) and np.isnan(r.precision)) else float(r.precision),
                            "choices": [str(c) for c in ch] if ch is not None else None, "answer": str(r.answer)}
    return meta, {"repo": MATHVISTA_REPO, "revision": sha, "files": [f.name for f in files]}


# ----------------------------------------------------------------------------- preparation
def prepare(df, name, mathvista_meta=None, log=print):
    """Add per-rollout answer group keys and correctness (columns: group, correct, no_answer)."""
    keys, corr, info = [], [], {"most_similar_fallback": 0, "normalisation_failed": 0, "gold_changed_by_normalisation": 0}
    parse = verify = None
    if name == "math":
        parse, verify = _mv()
    qs = df["sample_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    for n_done, s in enumerate(starts):
        sub = df.iloc[s:s + N_ROLLOUTS]
        ans = sub["extracted_answer"].tolist()
        if name == "math":
            k, c = math_groups_and_grades(ans, sub["gold_answer"].iloc[0], parse, verify)
        else:
            meta = mathvista_meta[str(sub["sample_id"].iloc[0])]
            k, c, st = mathvista_groups_and_grades(ans, meta)
            info["most_similar_fallback"] += st["most_similar_fallback"]
            info["normalisation_failed"] += st["normalisation_failed"]
            info["gold_changed_by_normalisation"] += int(meta["question_type"] != "multi_choice" and st["gold"] != meta["answer"])
        keys += k
        corr += c
        if (n_done + 1) % 200 == 0:
            log(f"  {name}: graded {n_done + 1} / {len(starts)} questions")
    out = df.copy()
    out["group"] = keys
    out["correct"] = np.array(corr, dtype=bool)
    out["no_answer"] = out["group"] == NO_ANSWER
    return out, info
