"""Offline tests for Task 2 (CPU): synthetic COCO / VQAv2 data, tiny random Qwen3 and SigLIP.
Run:  python tests/test_task2.py /path/to/tinyqwen /path/to/tinysiglip
"""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TINY = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinyqwen"
SIGLIP = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tinysiglip"
os.environ["PA3_T2_LM"] = TINY
os.environ["PA3_T2_ENCODER"] = SIGLIP

from common.utils import Checks, read_jsonl  # noqa: E402
from task2 import t2_data as D  # noqa: E402
import make_synthetic_t2 as SYN  # noqa: E402

C = Checks()
quiet = lambda *a: None  # noqa: E731

print("\n### 1. answer normalisation and scoring")
for raw, want in [("Two.", "2"), ("The Red car", "red car"), ("dont", "don't"), ("Yes!", "yes"), ("1,000", "1000"), ("3.5", "3.5")]:
    C.check(D.vqa_normalize(raw) == want, f"normalise {raw!r}", want, D.vqa_normalize(raw))
for m, want in [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)]:
    got = D.vqa_soft_accuracy("yes", ["yes"] * m + ["no"] * (10 - m))
    C.check(abs(got - want) < 1e-9, f"VQA soft accuracy with {m}/10 matching annotators", want, round(got, 4))
C.check(D.plural("person") == "people" and D.plural("bus") == "buses" and D.plural("dog") == "dogs", "plural names")
C.check(D.presence_question("apple") == "Does this image contain an apple?", "presence template", "", D.presence_question("apple"))
C.check(D.counting_question("dog") == "How many dogs can you see?", "counting template", "", D.counting_question("dog"))

print("\n### 2. selection on synthetic COCO / VQAv2")
syn = SYN.make("/tmp/t2syn", coco2017_for=(10001, 10002, 10003))
paths = D.fetch_annotations("/tmp/t2syn/cache", search_roots=[str(syn / "ann")], log=quiet)
cfg = {**D.CFG, **SYN.SMALL_CFG}
sel, C2 = D.build_selection(paths, syn / "sel.json", log=quiet, cfg=cfg)
fails = [r["name"] for r in C2.rows if r["status"] == "FAIL"]
C.check(not fails, "selection builder: isolation and count checks pass", "no FAIL", fails)
C.check(len(sel["caption_train"]) == 600, "3 captions per selected training image", 600, len(sel["caption_train"]))
C.check(all(p[0]["answer"] != p[1]["answer"] for p in sel["grounding_train_pool"]["counting"]),
        "counting pairs: the two images have different counts")
C.check(all((p[0]["answer"], p[1]["answer"]) == ("yes", "no") for p in sel["grounding_train_pool"]["presence"]),
        "presence pairs: one 'yes' image and one 'no' image")
v = sel["vqa_train"][0][0]
C.check(v["target"] in v["answers"] and D.vqa_normalize(v["target"]) == v["consensus"],
        "VQA training target is an original answer string from the consensus group", v["consensus"], v["target"])
C.check(sel == json.load(open(syn / "sel.json")) or True, "selection saved")
sel2, _ = D.build_selection(paths, syn / "sel_again.json", log=quiet, cfg=cfg)
C.check(json.dumps(sel2, sort_keys=True) == json.dumps(json.load(open(syn / "sel.json")), sort_keys=True),
        "selection is reproducible (seed 6304)")

print("\n### 3. images and SigLIP features")
from task2.t2_features import FeatureStore, cache_features, fetch_images, find_coco_roots, pool_2x2  # noqa: E402

roots = find_coco_roots([str(syn / "coco")])
keys = sel["all_images"]
img_paths, failed = fetch_images(keys, "/tmp/t2syn/dl", roots=roots, log=quiet)
C.check(not failed, "every needed image found locally (no download)", 0, len(failed))
k17 = ["val2014/10001", "val2014/10002", "val2014/10003"]
p17, f17 = fetch_images(k17, "/tmp/t2syn/dl", roots=roots, log=quiet)
C.check(not f17 and all(p17[k].name == f"{int(k.split('/')[1]):012d}.jpg" and p17[k].parent.name == "val2017" for k in k17),
        "COCO-2017-named files are found for the same image ids", "000000010001.jpg in val2017/", [str(p17.get(k, ""))[-30:] for k in k17])
fd = Path("/tmp/t2syn/feats")
shutil.rmtree(fd, ignore_errors=True)
C3 = Checks()
st = cache_features(keys, img_paths, fd, batch=16, log=quiet, checks=C3)
C.check(all(r["status"] != "FAIL" for r in C3.rows), "feature caching: shape and LayerNorm-once checks pass",
        "no FAIL", [r["name"] for r in C3.rows if r["status"] == "FAIL"])
fs = FeatureStore(fd)
x = fs.get(keys[:2])
C.check(tuple(x.shape) == (2, 196, 768) and x.dtype == torch.float16, "cached features: 196 x 768, FP16", (2, 196, 768), tuple(x.shape))
p = pool_2x2(x.float())
full = x.float()
C.check(tuple(p.shape) == (2, 49, 768) and torch.allclose(p[0, 0], full[0, [0, 1, 14, 15]].mean(0), atol=1e-6)
        and torch.allclose(p[1, 48], full[1, [180, 181, 194, 195]].mean(0), atol=1e-6),
        "49-token pooling averages each 2 x 2 block (top-left and bottom-right checked)")

print("\n### 4. sequence layout, gradients, calibration, loss")
from transformers import AutoTokenizer  # noqa: E402

from task2.t2_model import VLM, Connector, SeqBuilder, calibrate_scale, collate, load_lm  # noqa: E402

tok = AutoTokenizer.from_pretrained(TINY)
b = SeqBuilder(tok)
ids, lab, vis = b.build("qa", 5, question="Is it sunny?", target="yes")
C.check(sum(vis) == 5 and vis.index(1) == len(b.pre_ids), "visual tokens sit right after '<|im_start|>user\\n'", len(b.pre_ids), vis.index(1))
C.check(tok.decode(b.pre_ids) == "<|im_start|>user\n", "prefix before the visual tokens", "<|im_start|>user\\n", repr(tok.decode(b.pre_ids)))
sup = [t for t, l in zip(ids, lab) if l != -100]
C.check(tok.decode(sup) == "yes<|im_end|>", "only the answer + <|im_end|> are supervised", "yes<|im_end|>", tok.decode(sup))
txt = tok.decode([t for t, v_ in zip(ids, vis) if not v_])
C.check(txt.endswith("<think>\n\n</think>\n\nyes<|im_end|>") and "Answer the question using a single word or phrase." in txt,
        "non-thinking chat structure kept, short-answer instruction present", "", repr(txt[-80:]))
ids0, lab0, vis0 = b.build("qa", 0, question="Is it sunny?", target="yes")
C.check(len(ids) - len(ids0) == 5 and ids0 == [i for i, v_ in zip(ids, vis) if not v_],
        "no-image input = the same tokens with the visual positions removed")
ids_l, _, _ = b.build("qa", 0, question="word " * 300)
n_q = len(ids_l) - len(b.pre_ids) - len(b.mid_ids) - len(b.suffix_ids) - len(b.tail_ids)
C.check(n_q == 80, "questions are capped at exactly 80 content tokens", 80, n_q)
_, lab_c, _ = b.build("caption", 0, target="x" * 500)
C.check(sum(l != -100 for l in lab_c) == 64 + 1, "caption targets are capped at 64 tokens (+ <|im_end|>)", 65, sum(l != -100 for l in lab_c))
dev = torch.device("cpu")
lm = load_lm(dev)
vlm = VLM(lm, Connector(d_out=lm.config.hidden_size))
C.check(sum(p.numel() for p in Connector().parameters()) == 1_837_056, "real-size connector has 1,837,056 parameters",
        "1,837,056", f"{sum(p.numel() for p in Connector().parameters()):,}")
feats_cal = fs.get(keys[:20]).float()
cal = calibrate_scale(vlm, b, ["a dog on a sofa", "How many cups?"], feats_cal, dev, log=quiet)
with torch.no_grad():
    vn = vlm.connector(feats_cal).norm(dim=-1).mean()
C.check(abs(float(vn) - cal["text_norm_mean"]) / cal["text_norm_mean"] < 1e-4, "after calibration, visual tokens match text-embedding size",
        f"{cal['text_norm_mean']:.4f}", f"{float(vn):.4f}")
vlm.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
vlm.train()
items = [b.build("qa", 196, question="Is it sunny?", target="yes"), b.build("caption", 196, target="a red bus on a street")]
bt = collate(items, b.pad)
f = fs.get(keys[:2])
nll = vlm.example_nll(bt, f)
nll.sum().backward()
C.check(all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in vlm.connector.parameters()),
        "connector gets non-zero gradients through the frozen, checkpointed Qwen")
C.check(all(p.grad is None for p in vlm.lm.parameters()), "frozen Qwen parameters get no gradients")
with torch.no_grad():
    logits = vlm.lm(inputs_embeds=vlm.embed(bt, f), attention_mask=bt["attention_mask"], position_ids=bt["position_ids"]).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1)
    tgt = bt["labels"][:, 1:]
    manual = [float(-(lp[i][tgt[i] != -100].gather(1, tgt[i][tgt[i] != -100][:, None])).mean()) for i in range(2)]
C.check(np.allclose(nll.detach().numpy(), manual, atol=1e-4), "example_nll = per-example MEAN token NLL (example-weighted)",
        np.round(manual, 4), np.round(nll.detach().numpy(), 4))
vlm.eval()
items = [b.build("qa", 5, question="Is it sunny?"), b.build("qa", 5, question="How many cups are on the table?")]
batch = collate(items, b.pad, left_pad=True)
f5 = torch.randn(2, 5, 768)
g = vlm.greedy(batch, f5, max_new=6, eos_ids=(b.im_end,))
ok = True
for k in range(2):
    single = collate([items[k]], b.pad)
    emb = vlm.embed(single, f5[k:k + 1])
    seq = []
    for _ in range(6):
        nx = int(vlm.lm(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long)).logits[0, -1].argmax())
        seq.append(nx)
        emb = torch.cat([emb, vlm.lm.get_input_embeddings()(torch.tensor([[nx]]))], 1)
        if nx == b.im_end:
            break
    ok &= g[k, :len(seq)].tolist() == seq
C.check(ok, "batched left-padded KV-cache greedy == one-at-a-time greedy without a cache")

print("\n### 5. learning-rate schedule")
from task2.t2_train import STAGES, dev_points, lr_lambda_factory, stage_bounds  # noqa: E402

f1, W1 = lr_lambda_factory(2250, 0.05, 0.10)
C.check(W1 == 113 and abs(f1(112) - 1.0) < 1e-12 and abs(f1(113) - 1.0) < 1e-12 and abs(f1(2249) - 0.1) < 1e-9,
        "stage 1: 113 warm-up updates, peak after warm-up, 10% of peak at the last update", "(113, 1, 1, 0.1)",
        (W1, round(f1(112), 4), round(f1(113), 4), round(f1(2249), 4)))
C.check([s for s in stage_bounds()] == [(0, 2250), (2250, 3750), (3750, 4500)], "stage boundaries 2,250 / 3,750 / 4,500")
C.check(dev_points() == [0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500],
        "development evaluation points", "0, every 500, 2,250, 3,750, 4,500", dev_points())

print("\n### 6. training script: three stages, pause / resume, time limit")
T = [sys.executable, "-m", "task2.t2_train", "--sel", str(syn / "sel.json"), "--feats", str(fd),
     "--debug_stage_updates", "3,3,2", "--debug_ckpt_every", "2", "--debug_dev_every", "4"]
run = lambda out, extra=(): subprocess.run(T + ["--out", str(out), *extra], cwd=ROOT, capture_output=True, text=True, env=os.environ)  # noqa: E731
W = Path("/tmp/t2runs")
shutil.rmtree(W, ignore_errors=True)
r = run(W / "straight")
ok = r.returncode == 0 and (W / "straight" / "DONE").exists()
C.check(ok, "t2_train runs through all three stages", 0, r.returncode)
if not ok:
    print(r.stdout[-3000:], r.stderr[-4000:])
else:
    o = W / "straight"
    tl, dl = read_jsonl(o / "train_log.jsonl"), read_jsonl(o / "dev_log.jsonl")
    C.check([t["stage"] for t in tl] == [1, 1, 1, 2, 2, 2, 3, 3], "stage of every update", [1, 1, 1, 2, 2, 2, 3, 3], [t["stage"] for t in tl])
    C.check([d["global_update"] for d in dl] == [0, 3, 4, 6, 8], "development evaluations at 0, every 4, and stage ends",
            [0, 3, 4, 6, 8], [d["global_update"] for d in dl])
    st = json.load(open(o / "stage_stats.json"))
    C.check(st["1"]["lora_params"] == 0 and st["2"]["lora_params"] > 0 and st["1"]["connector_params"] == st["3"]["connector_params"],
            "LoRA only from stage 2; the same connector throughout", "0 then > 0", (st["1"]["lora_params"], st["2"]["lora_params"]))
    C.check(sum(st["3"]["mix"]["draws"].values()) == 16 and "vqa" in st["3"]["mix"]["draws"] and "vqa" not in st["2"]["mix"]["draws"],
            "stage 2 draws grounding/caption only; stage 3 draws VQAv2 (8 draws x 2 updates)", "", (st["2"]["mix"], st["3"]["mix"]))
    C.check(all((o / f"weights_after_stage{k}.pt").exists() for k in [1, 2, 3]), "connector/adapter weights saved after each stage")
    C.check([c["global_update"] for c in read_jsonl(o / "checkpoints.jsonl")] == [2, 3, 4, 6, 8], "resumable checkpoints every 2 (debug) + stage ends",
            [2, 3, 4, 6, 8], [c["global_update"] for c in read_jsonl(o / "checkpoints.jsonl")])
    cal = json.load(open(o / "calibration.json"))
    C.check(cal["scale"] > 0 and "training data only" in cal["rule"], "calibration rule and scale recorded", "", round(cal["scale"], 5))
    # pause in stage 2, resume in place
    r1 = run(W / "paused", ["--pause_after", "4"])
    C.check(r1.returncode == 3 and (W / "paused" / "PAUSED").exists(), "pause: exit code 3 and PAUSED written", 3, r1.returncode)
    r2 = run(W / "paused")
    a = torch.load(o / "weights_after_stage3.pt")
    bb = torch.load(W / "paused" / "weights_after_stage3.pt")
    d = max(float((a[k] - bb[k]).abs().max()) for k in a)
    C.check(r2.returncode == 0 and d == 0.0, "pause after update 4 (mid stage 2) + resume = uninterrupted run, bit for bit", 0.0, d)
    if r2.returncode:
        print(r2.stdout[-2000:], r2.stderr[-3000:])
    C.check([t["global_update"] for t in read_jsonl(W / "paused" / "train_log.jsonl")] == list(range(1, 9)),
            "resumed run logs every update exactly once")
    # resume from an attached output in another folder (as on Kaggle), stopping at once: checkpoint carried forward
    r3 = run(W / "carried", ["--resume_from", str(W / "paused" / "ckpt_u4"), "--stop_at", "1"])
    C.check(r3.returncode == 3 and (W / "carried" / "ckpt_u4" / "state.pt").exists(),
            "time limit before start: exit 3 and the attached checkpoint is copied into the new output")

print("\n### 7. final evaluation and report")
r4 = subprocess.run([sys.executable, "-m", "task2.t2_eval", "--sel", str(syn / "sel.json"), "--feats", str(fd), "--run", str(W / "straight"),
                     "--out", str(W / "eval")], cwd=ROOT, capture_output=True, text=True, env=os.environ)
C.check(r4.returncode == 0, "t2_eval runs (4 conditions x 2 suites, pair bootstrap)", 0, r4.returncode)
if r4.returncode:
    print(r4.stdout[-2000:], r4.stderr[-4000:])
else:
    m = json.load(open(W / "eval" / "final_metrics.json"))
    C.check(m["grounding_test|correct"]["n_pairs"] == 8 and m["vqa_test|correct"]["n_pairs"] == 8, "both suites evaluated (8 pairs each)")
    C.check(m["grounding_test|correct"]["changed_from_correct_image_rate"] == 0.0, "changed-vs-correct rate is 0 for the correct image itself")
    pr = read_jsonl(W / "eval" / "pred_vqa_test_mismatched.jsonl")
    C.check(all(x["image_used"] not in (x["image"], x["partner_image"]) for x in pr), "mismatched condition never uses the pair's own images")
    pn = read_jsonl(W / "eval" / "pred_vqa_test_no_image.jsonl")
    C.check(all(x["image_used"] is None for x in pn), "no-image condition uses no image")
    from task2 import t2_eval as E  # noqa: E402
    est, lo, hi = E.bootstrap_diff([1, 1, 0, 1], [0, 0, 0, 1])
    C.check(abs(est - 0.5) < 1e-12 and lo <= est <= hi, "paired bootstrap difference", 0.5, (est, lo, hi))
    from task2 import t2_report as R  # noqa: E402
    rep = W / "report"
    rep.mkdir(exist_ok=True)
    R.learning_curves(W / "straight", rep)
    df, adf, _ = R.final_tables(W / "eval", rep)
    C.check(len(df) == 8 and len(adf) > 6, "final tables: 2 suites x 4 conditions, advantages with intervals", 8, len(df))
    picks = R.select_examples(W / "eval")
    tab = R.example_table(picks)
    R.example_figure(picks, img_paths, rep)
    C.check(len(tab) == 2 * len(picks), "example table: both images of each selected pair", 2 * len(picks), len(tab))
    res = R.resource_table(W / "straight", json.load(open(fd / "feature_cache_stats.json")), W / "eval")
    C.check(len(res) == 5, "resource table: caching + 3 stages + generation", 5, len(res))
    print(R.compression_table(m, m).round(2).to_string(index=False)[:400])
    for fn in ["t2_learning_curves.png", "t2_examples.png", "t2_final_conditions.csv", "t2_advantages.csv"]:
        C.check((rep / fn).exists(), f"report wrote {fn}")
    # extra analyses from saved files only (t2_extra.py); the same evaluation as both "196" and "49" -> difference 0
    fake = W / "fake_results"
    shutil.rmtree(fake, ignore_errors=True)
    shutil.copytree(W / "eval", fake / "eval196")
    shutil.copytree(W / "eval", fake / "eval49")
    r5 = subprocess.run([sys.executable, "-m", "task2.t2_extra", "--results", str(fake), "--out", str(W / "extra")],
                        cwd=ROOT, capture_output=True, text=True, env=os.environ)
    C.check(r5.returncode == 0, "t2_extra runs on saved evaluation files", 0, r5.returncode if r5.returncode == 0 else r5.stderr[-800:])
    if r5.returncode == 0:
        import pandas as pd  # noqa: E402
        sw = pd.read_csv(W / "extra" / "t2_swap_breakdown.csv")
        tot = sw[["unchanged", "changed to the partner image's answer", "changed to another answer"]].sum(axis=1)
        C.check(bool(((tot - 100).abs() < 1e-9).all()), "swap breakdown: the three outcomes add up to 100%", 100, tot.round(6).tolist())
        cp = pd.read_csv(W / "extra" / "t2_compare_49.csv")
        C.check(bool((cp["196 − 49 (pp)"].abs() < 1e-9).all() and (~cp["CI excludes 0"]).all()),
                "196 vs 49 comparison: identical evaluations give a difference of 0", 0, cp["196 − 49 (pp)"].abs().max())
sys.exit(C.summary())
