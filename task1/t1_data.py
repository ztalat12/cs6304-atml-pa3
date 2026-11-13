"""Task 1 data preparation.

1. Select the first 40,000 eligible MetaMathQA examples, in source order (no shuffling).
   Eligible = non-empty question and response, formatted length <= 2048 tokens, and neither
   `query` nor `original_question` matches a GSM8K test question after normalisation (decision D1).
2. Tokenise each example as  [prompt tokens][response tokens][<|im_end|>]  where the prompt is
   Qwen3's chat template for one user turn, enable_thinking=False, with the generation prompt.
   The loss is later computed only on the response tokens and the <|im_end|> token.
3. Draw the activation-probe set (32 GSM8K test questions + 32 HellaSwag validation contexts)
   with seed 6304, before any model results exist.
"""
from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

from common.utils import SEED, Checks, write_json

MODEL_ID = os.environ.get("PA3_T1_MODEL", "Qwen/Qwen3-0.6B")   # env override is only for offline unit tests
N_TRAIN = 40_000
MAX_LEN = 2048


def normalize_question(text: str) -> str:
    """Decision D1: Unicode NFKC -> lowercase -> every non-alphanumeric character becomes a space
    -> collapse runs of spaces. Two questions 'match' if their normalised strings are identical."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"[^0-9a-z]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_prompt(tok, question: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def hellaswag_query(doc) -> str:
    """Identical to lm-eval's hellaswag/utils.py preprocessing (the stock completion prompt)."""
    def pre(text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        return text.replace("  ", " ")
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return pre(doc["activity_label"] + ": " + ctx)


def gsm8k_cot_text(question: str) -> str:
    """lm-eval gsm8k_cot doc_to_text; with apply_chat_template it becomes the user message."""
    return f"Q: {question}\nA:"


def prepare(out_dir: str, checks: Checks | None = None, n_train=N_TRAIN, log=print):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    checks = checks or Checks()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    # ---- chat-format sanity: our concatenation must equal Qwen3's own rendering of a finished turn
    demo_q, demo_a = "What is 2+3?", "2+3 = 5. The answer is: 5"
    prompt = build_prompt(tok, demo_q)
    official = tok.apply_chat_template([{"role": "user", "content": demo_q}, {"role": "assistant", "content": demo_a}],
                                       tokenize=False, enable_thinking=False)
    ours = prompt + demo_a + "<|im_end|>"
    log("prompt exactly as the model sees it (thinking disabled):\n" + repr(prompt))
    checks.check(prompt.endswith("<think>\n\n</think>\n\n"),
                 "generation prompt contains the empty think block (thinking disabled)",
                 "prompt ends with '<think>\\n\\n</think>\\n\\n'", repr(prompt[-30:]))
    checks.check(official.startswith(ours),
                 "prompt + response + <|im_end|> matches Qwen3's own template for a finished assistant turn",
                 "official rendering starts with our string", f"official tail = {official[max(0, len(ours) - 20):]!r}")

    # ---- GSM8K test questions for the overlap filter
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_norm = {normalize_question(q) for q in gsm["question"]}
    checks.check(len(gsm) == 1319, "GSM8K test split size", 1319, len(gsm))

    # ---- MetaMathQA in source order
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    log(f"MetaMathQA rows: {len(mm):,}; columns: {mm.column_names}")
    counts = {"scanned": 0, "empty": 0, "gsm8k_overlap": 0, "too_long": 0}
    overlap_examples = []
    kept_rows, ids_chunks, offsets, prompt_lens, types = [], [], [0], [], []
    joint_mismatch, joint_checked = 0, 0
    for i, ex in enumerate(mm):
        counts["scanned"] += 1
        q, a = ex.get("query") or "", ex.get("response") or ""
        if not q.strip() or not a.strip():
            counts["empty"] += 1
            continue
        if normalize_question(q) in gsm_norm or normalize_question(ex.get("original_question") or "") in gsm_norm:
            counts["gsm8k_overlap"] += 1
            if len(overlap_examples) < 5:
                overlap_examples.append({"row": i, "query": q[:200]})
            continue
        p_text = build_prompt(tok, q)
        p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
        r_ids = tok(a, add_special_tokens=False)["input_ids"]
        seq = p_ids + r_ids + [im_end]
        if len(seq) > MAX_LEN:
            counts["too_long"] += 1
            continue
        if joint_checked < 1000:          # does tokenising the whole string give the same tokens?
            joint = tok(p_text + a + "<|im_end|>", add_special_tokens=False)["input_ids"]
            joint_mismatch += int(joint != seq)
            joint_checked += 1
        kept_rows.append(i)
        ids_chunks.append(np.asarray(seq, dtype=np.int32))
        offsets.append(offsets[-1] + len(seq))
        prompt_lens.append(len(p_ids))
        types.append(ex.get("type", ""))
        if len(kept_rows) == n_train:
            break
    ids = np.concatenate(ids_chunks)
    offsets = np.asarray(offsets, dtype=np.int64)
    prompt_lens = np.asarray(prompt_lens, dtype=np.int32)
    lengths = np.diff(offsets)
    resp_tokens = lengths - prompt_lens          # response tokens + <|im_end|> = the supervised positions
    np.savez_compressed(out / "t1_train_tokens.npz", ids=ids, offsets=offsets, prompt_lens=prompt_lens,
                        source_rows=np.asarray(kept_rows, dtype=np.int64))
    write_json(out / "t1_train_ids.json", {"source": "meta-math/MetaMathQA", "split": "train",
                                           "source_row_indices": kept_rows, "exclusion_counts": counts,
                                           "gsm8k_overlap_examples": overlap_examples,
                                           "normalisation": "NFKC, lowercase, non-alphanumeric->space, collapse spaces; exact match"})
    checks.check(len(kept_rows) == n_train, "selected exactly 40,000 eligible examples", n_train, len(kept_rows))
    checks.check(bool(np.all(np.diff(kept_rows) > 0)), "selected rows keep source order (strictly increasing)", True,
                 bool(np.all(np.diff(kept_rows) > 0)))
    checks.check(int(lengths.max()) <= MAX_LEN, "no formatted example longer than 2,048 tokens", "<= 2048", int(lengths.max()))
    checks.observe("exclusions while scanning to the 40,000th eligible row", counts)
    checks.observe("tokenising prompt and response separately vs as one string (first 1,000 kept)",
                   f"{joint_mismatch} of {joint_checked} differ",
                   note="0 means the prompt/response boundary never changes the tokens")
    checks.observe("token statistics", {"mean_len": round(float(lengths.mean()), 1), "p50": float(np.median(lengths)),
                                        "p99": float(np.percentile(lengths, 99)), "max": int(lengths.max()),
                                        "mean_supervised_tokens": round(float(resp_tokens.mean()), 1),
                                        "total_supervised_tokens": int(resp_tokens.sum())})
    tcount = {}
    for t in types:
        tcount[t] = tcount.get(t, 0) + 1
    checks.observe("MetaMathQA 'type' mix in the 40,000", tcount)
    per_update = resp_tokens.reshape(1250, 32).sum(1)
    checks.observe("supervised tokens per effective batch (the loss denominator)",
                   {"min": int(per_update.min()), "mean": round(float(per_update.mean()), 1), "max": int(per_update.max())})

    # ---- activation-probe set (drawn before any model results exist)
    rng = np.random.default_rng(SEED)
    hs = load_dataset("Rowan/hellaswag", split="validation")
    checks.check(len(hs) == 10042, "HellaSwag validation size", 10042, len(hs))
    gsm_idx = sorted(int(x) for x in rng.choice(len(gsm), 32, replace=False))
    hs_idx = sorted(int(x) for x in rng.choice(len(hs), 32, replace=False))
    probe = {"seed": SEED,
             "rule": "rng = np.random.default_rng(6304); 32 GSM8K test indices, then 32 HellaSwag validation "
                     "indices, each rng.choice(N, 32, replace=False), sorted",
             "gsm8k_test_idx": gsm_idx, "hellaswag_val_idx": hs_idx,
             "gsm8k": [{"idx": i, "text": gsm8k_cot_text(gsm[i]["question"])} for i in gsm_idx],
             "hellaswag": [{"idx": i, "ind": hs[i].get("ind"), "text": hellaswag_query(hs[i])} for i in hs_idx]}
    write_json(out / "t1_probe_set.json", probe)
    checks.check(len(set(gsm_idx)) == 32 and len(set(hs_idx)) == 32, "probe set has 32 + 32 distinct items",
                 "32 + 32", f"{len(set(gsm_idx))} + {len(set(hs_idx))}")
    log(f"data preparation finished in {time.time() - t0:.0f}s -> {out}")
    return checks


class TrainData:
    """Deterministic microbatch provider. Update u (0-based) uses examples 32u .. 32u+31 in source order;
    microbatch j of that update uses examples 32u+2j and 32u+2j+1."""

    def __init__(self, npz_path, micro=2, accum=16):
        d = np.load(npz_path)
        self.ids, self.offsets, self.prompt_lens = d["ids"], d["offsets"], d["prompt_lens"]
        self.micro, self.accum = micro, accum
        self.eff = micro * accum
        self.n = len(self.prompt_lens)
        lengths = np.diff(self.offsets)
        self.sup = lengths - self.prompt_lens

    def n_updates(self):
        return self.n // self.eff

    def update_token_count(self, u):
        return int(self.sup[u * self.eff:(u + 1) * self.eff].sum())

    def microbatch(self, u, j, pad_id):
        import torch
        start = u * self.eff + j * self.micro
        seqs, plens = [], []
        for e in range(start, start + self.micro):
            seqs.append(self.ids[self.offsets[e]:self.offsets[e + 1]])
            plens.append(int(self.prompt_lens[e]))
        L = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        labels = torch.full((len(seqs), L), -100, dtype=torch.long)
        for b, (s, pl) in enumerate(zip(seqs, plens)):
            s = torch.from_numpy(s.astype(np.int64))
            input_ids[b, :len(s)] = s
            attn[b, :len(s)] = 1
            labels[b, pl:len(s)] = s[pl:]          # response tokens + <|im_end|>; prompt and padding stay -100
        return input_ids, attn, labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/t1_data")
    a = ap.parse_args()
    c = prepare(a.out)
    c.summary()
    c.save(Path(a.out) / "t1_data_checks.json")
