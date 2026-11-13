"""Synthetic stand-in for ATML-PA3-Data (same folders, files, columns and list alignment as the supplied README.MD),
small enough for offline CPU tests. Usage: python tests/make_synthetic_t3.py OUT_DIR [N_QUESTIONS]
Writes OUT_DIR/ATML-PA3-Data/{qwen25-math, qwen25vl-mathvista}/*.parquet, OUT_DIR/mathvista_meta.json and
OUT_DIR/readme_totals.json (the totals the README check should expect for this synthetic data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

R = 32


def lists(rng, n, lo, hi):
    return [rng.uniform(lo, hi, size=n).astype(np.float32).tolist()]


def make(out, nq=60, seed=0, vlm_tokens=(1, 30)):
    out = Path(out)
    rng = np.random.default_rng(seed)
    root = out / "ATML-PA3-Data"
    totals = {}
    # ------------------------------------------------------------------ MATH
    golds = ["\\frac{1}{2}", "3", "2\\sqrt{3}", "10", "x^2+1", "\\dfrac{3}{4}"]
    equiv = {"\\frac{1}{2}": ["0.5", "\\frac12", "1/2"], "3": ["3", "3.0"], "2\\sqrt{3}": ["\\sqrt{12}", "2\\sqrt3"],
             "10": ["10", "10.0"], "x^2+1": ["x^2+1", "1+x^2"], "\\dfrac{3}{4}": ["0.75", "\\frac{3}{4}"]}
    G, P = {k: [] for k in ["dataset", "sample_id", "rollout_id", "model", "question_text", "generated_text", "generated_token_ids",
                            "n_generated_text_tokens", "hit_max_new_tokens", "gold_answer", "extracted_answer"]}, \
        {k: [] for k in ["dataset", "sample_id", "rollout_id", "logp", "kl_u_p"]}
    for q in range(nq):
        gold = golds[q % len(golds)]
        p_correct = rng.uniform(0, 1)
        for r in range(R):
            u = rng.uniform()
            if u < 0.08:
                ext = None
            elif u < 0.08 + 0.8 * p_correct:
                ext = str(rng.choice(equiv[gold]))
            else:
                ext = str(rng.integers(0, 6))
            ok = ext is not None and ext in equiv[gold]
            n = int(rng.integers(0, 40)) if r != 5 else 0
            G["dataset"].append("math"); G["sample_id"].append(f"test:{q * 7}"); G["rollout_id"].append(r)
            G["model"].append("Qwen/Qwen2.5-3B-Instruct"); G["question_text"].append(f"Problem {q}")
            G["generated_text"].append(f"<think>steps</think> so \\boxed{{{ext}}}" if ext else "no box")
            G["generated_token_ids"].append(rng.integers(0, 1000, size=n).tolist()); G["n_generated_text_tokens"].append(n)
            G["hit_max_new_tokens"].append(bool(rng.uniform() < 0.03)); G["gold_answer"].append(gold); G["extracted_answer"].append(ext)
            P["dataset"].append("math"); P["sample_id"].append(f"test:{q * 7}"); P["rollout_id"].append(r)
            P["logp"].append(rng.uniform(-2, 0, size=n).astype(np.float32).tolist())
            P["kl_u_p"].append((rng.uniform(5, 12, size=n) + 1.5 * ok).astype(np.float32).tolist())
    d = root / "qwen25-math"
    d.mkdir(parents=True, exist_ok=True)
    perm = rng.permutation(len(G["dataset"]))      # store rows shuffled: the loader must join on keys, not rely on order
    pq.write_table(pa.table({k: [v[i] for i in perm] for k, v in G.items()}), d / "generations.parquet")
    pq.write_table(pa.table(P), d / "probability_metrics.parquet")
    totals["math"] = {"retained_tokens": int(sum(G["n_generated_text_tokens"])), "capped": int(sum(G["hit_max_new_tokens"])),
                      "null_extracted": int(sum(e is None for e in G["extracted_answer"]))}
    # ------------------------------------------------------------------ MathVista
    meta = {}
    G2 = {k: [] for k in G}
    P2 = {k: [] for k in P}
    A = {k: [] for k in ["dataset", "sample_id", "rollout_id", "n_image_tokens", "n_prompt_text_tokens", "img_attn_total",
                         "prompt_text_attn_total", "generated_text_attn_total", "n_generated_text_tokens"]}
    for q in range(nq):
        pid = str(q + 1)
        kind = q % 4
        if kind == 0:
            m = {"question_type": "multi_choice", "answer_type": "text", "precision": None,
                 "choices": ["red", "green", "blue", "yellow"], "answer": "blue"}
            right, wrong = ["C", "(C) blue", "blue"], ["A", "(B)", "green", "purple"]
        elif kind == 1:
            m = {"question_type": "free_form", "answer_type": "integer", "precision": None, "choices": None, "answer": "7"}
            right, wrong = ["7", "7.0"], ["8", "six", "3"]
        elif kind == 2:
            m = {"question_type": "free_form", "answer_type": "float", "precision": 2.0, "choices": None, "answer": "1.25"}
            right, wrong = ["1.25", "1.250"], ["1.2", "2.5"]
        else:
            m = {"question_type": "free_form", "answer_type": "list", "precision": None, "choices": None, "answer": "[1, 2]"}
            right, wrong = ["[1, 2]"], ["[2, 1]", "[1]"]
        meta[pid] = m
        p_correct = rng.uniform(0, 1)
        for r in range(R):
            u = rng.uniform()
            ext = None if u < 0.15 else (str(rng.choice(right)) if u < 0.15 + 0.75 * p_correct else str(rng.choice(wrong)))
            ok = ext in right
            n = int(rng.integers(*vlm_tokens))
            for D in (G2, P2, A):
                D["dataset"].append("mathvista"); D["sample_id"].append(pid); D["rollout_id"].append(r)
            G2["model"].append("Qwen/Qwen2.5-VL-3B-Instruct"); G2["question_text"].append(f"Q {pid}")
            G2["generated_text"].append(f"answer {ext}"); G2["generated_token_ids"].append(rng.integers(0, 1000, size=n).tolist())
            G2["n_generated_text_tokens"].append(n); G2["hit_max_new_tokens"].append(bool(rng.uniform() < 0.03))
            G2["gold_answer"].append(m["answer"]); G2["extracted_answer"].append(ext)
            P2["logp"].append(rng.uniform(-2, 0, size=n).astype(np.float32).tolist())
            P2["kl_u_p"].append((rng.uniform(5, 12, size=n) + 1.0 * ok).astype(np.float32).tolist())
            img = rng.uniform(0.05, 0.4, size=n) + 0.1 * ok
            gen = rng.uniform(0.1, 0.5, size=n)
            gen[0] = 0.0
            A["n_image_tokens"].append(256); A["n_prompt_text_tokens"].append(40)
            A["img_attn_total"].append(img.astype(np.float32).tolist())
            A["prompt_text_attn_total"].append(rng.uniform(0.1, 0.4, size=n).astype(np.float32).tolist())
            A["generated_text_attn_total"].append(gen.astype(np.float32).tolist())
            A["n_generated_text_tokens"].append(list(range(n)))
    d = root / "qwen25vl-mathvista"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(G2), d / "generations.parquet")
    perm = rng.permutation(len(P2["dataset"]))
    pq.write_table(pa.table({k: [v[i] for i in perm] for k, v in P2.items()}), d / "probability_metrics.parquet")
    pq.write_table(pa.table(A), d / "attention_metrics.parquet")
    totals["mathvista"] = {"retained_tokens": int(sum(G2["n_generated_text_tokens"])), "capped": int(sum(G2["hit_max_new_tokens"])),
                           "null_extracted": int(sum(e is None for e in G2["extracted_answer"]))}
    json.dump({"meta": meta, "source": {"repo": "synthetic", "revision": "test"}}, open(out / "mathvista_meta.json", "w"))
    json.dump(totals, open(out / "readme_totals.json", "w"))
    return root, totals


if __name__ == "__main__":
    print(make(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60))
