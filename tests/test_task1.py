"""Offline unit tests for Task 1 (CPU, tiny random Qwen3).  Run:  python tests/test_task1.py /path/to/tinyqwen

What is tested (each prints PASS / FAIL with expected vs actual):
  1  LoRA / OPLoRA start from the unchanged model; LoRA-16 and OPLoRA-16 share bit-identical initial A
  2  OPLoRA constraints and Proposition 2;  rho_k ~ 0 for OPLoRA but not for plain LoRA
  3  merged weights reproduce unmerged adapter logits
  4  masked, chunked SFT loss == naive masked cross-entropy
  5  L2-SP penalty and its manual gradient == autograd
  6  microbatches supervise only response + <|im_end|>; loss normaliser; 1,250 updates
  7  training script end-to-end for full / l2sp / lora4 / lora16 / oplora
  8  resume reproduces an uninterrupted run exactly (LoRA with dropout, and Full FT)
  9  FP16 export reloads with tied embeddings and the trained weights
 10  lm-eval wrapper records exact stop reasons; GSM8K rows get the right truncation flags
 11  evaluation pieces: merge check, qualitative generation
 12  diagnostics, the FP64 Proposition-2 recheck, and report code run on the smoke-run outputs
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TINY = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinyqwen"
os.environ["PA3_T1_MODEL"] = TINY
os.environ["PA3_T1_K"] = "8"          # the tiny model has only 32 output directions, so protect 8 instead of 128

from common.lora import adapter_state, attach_exported_adapter, inject_lora, lora_modules, merge_adapter_into  # noqa: E402
from common.utils import Checks, read_jsonl  # noqa: E402

C = Checks()
torch.manual_seed(0)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
quiet = lambda *a: None  # noqa: E731


def tiny():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)


print("\n### 1. starting point and shared initialisation")
m0, m1, m2 = tiny(), tiny(), tiny()
x = torch.randint(10, 90, (2, 12))
ref = m0(x).logits
i1 = inject_lora(m1, TARGETS, 16, 32, 0.0, 6304, log=quiet)
i2 = inject_lora(m2, TARGETS, 16, 32, 0.0, 6304, protect_k=8, log=quiet)
C.check(torch.allclose(m1(x).logits, ref, atol=1e-6), "LoRA with B = 0 leaves the model unchanged")
C.check(torch.allclose(m2(x).logits, ref, atol=1e-6), "OPLoRA with B = 0 leaves the model unchanged")
C.check(i1["init_A_sha256"] == i2["init_A_sha256"], "LoRA-16 and OPLoRA-16 draw bit-identical initial A",
        "same fingerprint", (i1["init_A_sha256"][:12], i2["init_A_sha256"][:12]))
C.check(i1["n_modules"] == 28 * 7, "adapters on all 7 projections in all 28 layers", 196, i1["n_modules"])

print("\n### 2. OPLoRA constraints and Proposition 2")
mod = lora_modules(m2)["model.layers.0.mlp.down_proj"]
with torch.no_grad():
    mod.lora_B.normal_()
dW = mod.delta_weight()
W0 = mod.base.weight.float()
U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
k = 8
C.check(float((mod.Uk.T @ dW).norm() / dW.norm()) < 1e-5, "Uk^T dW = 0 (no update along protected output directions)",
        "< 1e-5", float((mod.Uk.T @ dW).norm() / dW.norm()))
C.check(float((dW @ mod.Vk).norm() / dW.norm()) < 1e-5, "dW Vk = 0 (protected input directions produce no update)",
        "< 1e-5", float((dW @ mod.Vk).norm() / dW.norm()))
err = max(float(((W0 + dW) @ Vh[i] - S[i] * U[:, i]).norm()) for i in range(k))
C.check(err < 1e-4, "Prop. 2: (W0 + dW) v_i = sigma_i u_i for the top-k triples", "< 1e-4", err)
rho = float((mod.Uk @ (mod.Uk.T @ dW)).norm() ** 2 / dW.norm() ** 2)
C.check(rho < 1e-10, "rho_k of an OPLoRA update is ~0", "~0", rho)
lm_ = lora_modules(m1)["model.layers.0.mlp.down_proj"]
with torch.no_grad():
    lm_.lora_B.copy_(mod.lora_B)
dWl = lm_.delta_weight()
Uk = U[:, :k]
rho_l = float((Uk @ (Uk.T @ dWl)).norm() ** 2 / dWl.norm() ** 2)
C.check(rho_l > 0.01, "rho_k of the matching plain-LoRA update is clearly > 0", "> 0.01", round(rho_l, 4))

# real-size precision: FP32 SVD bases of a 1,024 x 3,072 matrix are re-orthonormalised once, so the projection
# residual stays small even when raw B has drifted far into the protected directions (AdamW does this)
from common.lora import topk_bases  # noqa: E402
torch.manual_seed(1)
Wbig = torch.randn(1024, 3072) * 0.02
Ub, Vb, Sb = topk_bases(Wbig, 128)
U_svd = torch.linalg.svd(Wbig, full_matrices=False)[0][:, :128]
C.check(float((Ub.T @ Ub - torch.eye(128)).norm()) < 5e-6, "real size: re-orthonormalised Uk is perpendicular to ~FP32 precision",
        "< 5e-6", f"{float((Ub.T @ Ub - torch.eye(128)).norm()):.1e}")
C.check(float((Ub - U_svd).abs().max()) < 1e-5, "real size: re-orthonormalisation keeps each singular vector (and its sign)",
        "max change < 1e-5", f"{float((Ub - U_svd).abs().max()):.1e}")
Bfree = torch.randn(1024, 16)
Bfree = Bfree - Ub @ (Ub.T @ Bfree)
Bleak = Bfree + 15 * Ub @ torch.randn(128, 16) * Bfree.norm() / (128 * 16) ** 0.5
Bt = Bleak - Ub @ (Ub.T @ Bleak)
resid = float((Ub.T @ Bt).norm() / Bt.norm())
C.check(resid < 1e-4, "real size: constraint residual with raw B drifted 15x into the protected zone", "< 1e-4", f"{resid:.1e}")
C.check(float((Wbig @ Vb - Ub * Sb[:128]).norm() / (Ub * Sb[:128]).norm()) < 1e-5, "real size: W0 vᵢ = σᵢ uᵢ still holds for the stored bases")

print("\n### 3. merging")
for n, m in lora_modules(m2).items():
    with torch.no_grad():
        m.lora_B.normal_(std=0.05)
unmerged_logits = m2(x).logits
ad = adapter_state(m2, projected=True)
fresh = tiny()
merge_adapter_into(fresh, ad, 2.0, log=quiet)
C.check(torch.allclose(fresh(x).logits, unmerged_logits, atol=1e-4), "merging projected factors reproduces unmerged OPLoRA logits",
        "max diff < 1e-4", float((fresh(x).logits - unmerged_logits).abs().max()))
re_ = attach_exported_adapter(tiny(), ad, 2.0)
C.check(torch.allclose(re_(x).logits, unmerged_logits, atol=1e-4), "re-attaching exported factors as plain LoRA reproduces OPLoRA",
        "max diff < 1e-4", float((re_(x).logits - unmerged_logits).abs().max()))

print("\n### 4. masked, chunked SFT loss")
from task1.t1_train import displacement_cpu, l2sp_penalty_and_grad, sft_loss_sum  # noqa: E402

m = tiny()
ids = torch.randint(10, 90, (2, 20))
attn = torch.ones_like(ids)
attn[1, 15:] = 0
labels = ids.clone()
labels[:, :6] = -100
labels[1, 15:] = -100
lsum, n = sft_loss_sum(m, ids, attn, labels, chunk=7)
logits = m(input_ids=ids, attention_mask=attn).logits[:, :-1]
tgt = labels[:, 1:]
naive = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="sum")
C.check(abs(float(lsum) - float(naive)) < 1e-3 and n == int((tgt != -100).sum()),
        "chunked masked loss equals naive masked cross-entropy (sum and token count)",
        f"{float(naive):.4f}, n = {int((tgt != -100).sum())}", f"{float(lsum):.4f}, n = {n}")
lsum.backward()
C.check(m.lm_head.weight.grad is not None, "gradients flow through the checkpointed loss chunks")

print("\n### 5. L2-SP penalty and gradient")
m = tiny()
params = list(m.parameters())
refs = [p.detach().to(torch.bfloat16).clone() for p in params]
C.check(all(torch.equal(r.float(), p.detach()) for r, p in zip(refs, params)), "BF16 copy of theta0 is bit-exact")
with torch.no_grad():
    for p in params:
        p.add_(torch.randn_like(p) * 1e-3)
for p in params:
    p.grad = torch.zeros_like(p)
lam = 1e-3
omega = l2sp_penalty_and_grad(params, refs, lam, add_grad=True, chunk=1000)
m2_ = tiny()
p2 = list(m2_.parameters())
with torch.no_grad():
    for a, b in zip(p2, params):
        a.copy_(b)
om = 0.5 * sum(((a - r.float()) ** 2).sum() for a, r in zip(p2, refs))
(lam * om).backward()
gdiff = max(float((a.grad - b.grad).abs().max()) for a, b in zip(p2, params))
C.check(abs(omega - float(om)) / float(om) < 1e-5, "Omega = 1/2 * sum (theta - theta0)^2 over all unique params",
        f"{float(om):.6g}", f"{omega:.6g}")
C.check(gdiff < 1e-9, "manual L2-SP gradient equals autograd of lambda * Omega", "< 1e-9", gdiff)
disp = displacement_cpu(m, {n: r for (n, _), r in zip(m.named_parameters(), refs)})
C.check(abs(disp - (2 * omega) ** 0.5) < 1e-6, "displacement = sqrt(2 * Omega)", f"{(2 * omega) ** 0.5:.6f}", f"{disp:.6f}")
C.observe("tied embedding counted once", f"{len(params)} unique parameter tensors vs {len(m.state_dict())} state-dict keys")

print("\n### 6. microbatches")
from task1.t1_data import TrainData  # noqa: E402

tmp = Path("/tmp/t1test")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir()
rng = np.random.default_rng(0)
lens = rng.integers(8, 20, 40000)
pl = (lens * 0.4).astype(np.int32)
offs = np.concatenate([[0], np.cumsum(lens)])
np.savez_compressed(tmp / "t1_train_tokens.npz", ids=rng.integers(10, 90, offs[-1]).astype(np.int32),
                    offsets=offs, prompt_lens=pl, source_rows=np.arange(40000))
td = TrainData(tmp / "t1_train_tokens.npz")
i_, a_, l_ = td.microbatch(3, 5, pad_id=0)
e0 = 3 * 32 + 10
C.check(int((l_[0] != -100).sum()) == lens[e0] - pl[e0], "microbatch: supervised positions = response + <|im_end|> only",
        int(lens[e0] - pl[e0]), int((l_[0] != -100).sum()))
C.check(td.update_token_count(3) == int((lens[96:128] - pl[96:128]).sum()), "loss normaliser = supervised tokens over the 32-example batch")
C.check(td.n_updates() == 1250, "1,250 optimizer updates from 40,000 examples", 1250, td.n_updates())

print("\n### 7. training script end-to-end")


def train(meth, out, n, extra=()):
    return subprocess.run([sys.executable, "-m", "task1.t1_train", "--method", meth, "--data", str(tmp), "--out", str(out),
                           "--max_updates", str(n), *extra], cwd=ROOT, capture_output=True, text=True, env=os.environ)


runs = Path("/tmp/t1runs")
shutil.rmtree(runs, ignore_errors=True)
ok_all = True
for meth in ["full", "l2sp", "lora4", "lora16", "oplora"]:
    r = train(meth, runs, 3, ["--keep_all_recovery"] if meth == "full" else [])
    ok = r.returncode == 0 and (runs / meth / "DONE").exists()
    ok_all &= ok
    if not ok:
        print(r.stdout[-3000:], r.stderr[-3000:])
        continue
    rows = read_jsonl(runs / meth / "train_log.jsonl")
    C.check(len(rows) == 3 and all(np.isfinite(x["sft_loss"]) for x in rows), f"{meth}: 3 finite log rows", 3, len(rows))
    if meth == "l2sp":
        C.check(rows[0]["omega"] == 0.0, "l2sp: Omega is exactly 0 before the first update", 0.0, rows[0]["omega"])
        C.check(rows[-1]["omega"] > 0, "l2sp: Omega grows once weights move", "> 0", rows[-1]["omega"])
    if meth == "full":
        C.check([round(x["lr"] / 2e-5 * 38, 6) for x in rows] == [0.0, 1.0, 2.0],
                "full: LR follows the warm-up (0, 1/38, 2/38 of the peak)", [0, 1, 2], [round(x["lr"] / 2e-5 * 38, 6) for x in rows])
fs = json.load(open(runs / "lora16" / "final_summary.json"))
C.check(fs["trainable_params"] == 16 * 28 * (2 * (32 + 32) + 2 * (32 + 16) + 3 * (32 + 48)) // 1,
        "lora16: trainable params = r x sum(din + dout) over all targets", "formula", fs["trainable_params"])
C.check(ok_all, "training script runs end-to-end for all five methods")
for meth in ["l2sp", "lora4", "lora16", "oplora"]:
    left = sorted(p.name for p in (runs / meth).glob("recovery_u*"))
    ck = read_jsonl(runs / meth / "checkpoints.jsonl")
    C.check(not left and ck and ck[-1]["recovery_bytes"] > 0,
            f"{meth}: finished run deletes its recovery checkpoints but keeps their logged sizes", "none left, size > 0",
            (left, ck[-1]["recovery_bytes"] if ck else None))
C.check(not (runs / "l2sp" / "export_mid").exists() and (runs / "l2sp" / "export_final").exists(),
        "l2sp: only the final model is exported (the midpoint is evaluated for Full FT only)")
for meth in ["lora4", "oplora"]:
    C.check((runs / meth / "adapter_final_projected.pt").exists(), f"{meth}: final adapter kept for evaluation")

print("\n### 8. pause and resume")
res = Path("/tmp/t1resume")
shutil.rmtree(res, ignore_errors=True)
for meth, fname in [("lora4", "adapter_final_projected.pt"), ("full", "diag_down_proj_final.pt"), ("oplora", "adapter_final_projected.pt")]:
    train(meth, res / "straight", 4)
    r1 = train(meth, res / "part", 4, ["--pause_after", "2"])
    paused = (res / "part" / meth / "PAUSED").exists() and not (res / "part" / meth / "DONE").exists()
    C.check(r1.returncode == 3 and paused, f"{meth}: a paused run exits with code 3, writes PAUSED, not DONE",
            "exit 3, PAUSED", (r1.returncode, paused))
    C.check((res / "part" / meth / "recovery_u2" / "state.pt").exists(), f"{meth}: pause saved a recovery checkpoint at update 2")
    if meth == "full":     # resume from an "attached" copy in another folder, as on Kaggle
        r2 = train(meth, res / "resumed", 4, ["--resume_from", str(res / "part" / meth / "recovery_u2")])
        out = res / "resumed"
    else:                  # resume in place, finding its own newest recovery checkpoint
        r2 = train(meth, res / "part", 4)
        out = res / "part"
    a = torch.load(res / "straight" / meth / fname)
    b = torch.load(out / meth / fname)
    dmax = max(float((a[k] - b[k]).abs().max()) for k in a)
    C.check(dmax == 0.0, f"{meth}: pause at update 2 + resume gives bit-identical weights at update 4", 0.0, dmax)
    if dmax != 0.0 or r2.returncode != 0:
        print(r2.stdout[-2000:], r2.stderr[-2000:])
    ups = [x["update"] for x in read_jsonl(out / meth / "train_log.jsonl")]
    C.check(ups == [1, 2, 3, 4], f"{meth}: resumed log has each update exactly once", [1, 2, 3, 4], ups)
    cks = [x["update"] for x in read_jsonl(out / meth / "checkpoints.jsonl")]
    C.check(cks == [4], f"{meth}: the pause checkpoint is not logged as a scheduled checkpoint", [4], cks)
    C.check(not (out / meth / "PAUSED").exists() and (out / meth / "DONE").exists(), f"{meth}: PAUSED cleared once finished")
# hourly safety checkpoints (here: after every update) are resumable but not logged as scheduled ones
r4 = train("lora4", res / "safety", 4, ["--safety_minutes", "0", "--keep_all_recovery"])
saved = sorted(int(p.name.split("_u")[1]) for p in (res / "safety" / "lora4").glob("recovery_u*"))
logged = [x["update"] for x in read_jsonl(res / "safety" / "lora4" / "checkpoints.jsonl")]
C.check(r4.returncode == 0 and saved == [1, 2, 3, 4] and logged == [4], "safety checkpoints: saved after updates 1-3, only update 4 logged",
        "saved [1, 2, 3, 4], logged [4]", (saved, logged))
# resuming from an attached output copies its checkpoint into the new output before doing anything else
r5 = train("lora4", res / "carried", 4, ["--resume_from", str(res / "safety" / "lora4" / "recovery_u2"), "--pause_after", "0"])
C.check(r5.returncode == 3 and (res / "carried" / "lora4" / "recovery_u2" / "state.pt").exists(),
        "resume from an attached output: its checkpoint is copied into the new output first", "exit 3 + recovery_u2",
        (r5.returncode, (res / "carried" / "lora4" / "recovery_u2" / "state.pt").exists()))
# a run whose time limit has already passed must exit at once, carrying its newest state into the new output
train("lora4", res / "late_part", 4, ["--pause_after", "1"])
r3 = subprocess.run([sys.executable, "-m", "task1.t1_train", "--method", "lora4", "--data", str(tmp), "--out", str(res / "late"),
                     "--max_updates", "4", "--stop_at", "1", "--resume_from", str(res / "late_part" / "lora4" / "recovery_u1")],
                    cwd=ROOT, capture_output=True, text=True, env=os.environ)
C.check(r3.returncode == 3 and (res / "late" / "lora4" / "recovery_u1" / "state.pt").exists(),
        "time limit before start: exit 3 and the attached state is carried into the new output", "exit 3 + recovery_u1",
        (r3.returncode, (res / "late" / "lora4" / "recovery_u1" / "state.pt").exists()))

print("\n### 9. FP16 export")
from transformers import AutoModelForCausalLM  # noqa: E402

ex = AutoModelForCausalLM.from_pretrained(runs / "full" / "export_final", dtype=torch.float16)
st = torch.load(runs / "full" / "recovery_u3" / "state.pt", weights_only=False)["trainable"]
C.check(ex.lm_head.weight.data_ptr() == ex.model.embed_tokens.weight.data_ptr(), "export keeps the embedding / output layer tied")
dmax = max(float((ex.state_dict()[k].float() - v.half().float()).abs().max()) for k, v in st.items() if k in ex.state_dict())
C.check(dmax == 0.0, "export = trained FP32 weights cast to FP16", 0.0, dmax)

print("\n### 10. lm-eval wrapper and GSM8K rows")
from lm_eval.api.instance import Instance  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from task1.t1_eval import gsm8k_rows, make_tracking_lm  # noqa: E402

tok = AutoTokenizer.from_pretrained(TINY)
model = tiny().eval()
lm = make_tracking_lm(model, tok, batch_size=4)
ctxs = [tok.apply_chat_template([{"role": "user", "content": f"Q: what is {i}+1?\nA:"}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False) for i in range(3)]
reqs = [Instance(request_type="generate_until", doc={}, arguments=(c, {"until": ["Q:"], "max_gen_toks": 6, "do_sample": False}),
                 idx=0) for c in ctxs]
outs = lm.generate_until(reqs)
C.check(all(c in lm.gen_log for c in ctxs), "every context is found in the generation log (exact string match)", 3,
        sum(c in lm.gen_log for c in ctxs))
info = lm.gen_log[ctxs[0]]
C.observe("generation log entry", {k: info[k] for k in ["n_raw", "hit_eos", "cap"]})
samples = []
for i, (c, o) in enumerate(zip(ctxs, outs)):
    for filt, ext in [("flexible-extract", "2"), ("strict-match", "[invalid]")]:
        samples.append({"doc_id": i, "doc": {"question": f"q{i}"}, "target": "2", "resps": [[o]], "filtered_resps": [ext],
                        "filter": filt, "exact_match": float(ext == "2"), "arguments": [(c, {})]})
rows, n_tr = gsm8k_rows(samples, tok, lm.gen_log, cap=6)
C.check(n_tr == 3, "all three GSM8K rows matched to tracked generations", 3, n_tr)
want = [(not lm.gen_log[c]["hit_eos"]) and lm.gen_log[c]["n_raw"] >= 6 and not any(t in lm.gen_log[c]["raw_text"] for t in ["Q:"])
        for c in ctxs]
C.check([r["truncated_at_cap"] for r in rows] == want, "truncation flag = no end-of-turn, no stop string, cap reached", want,
        [r["truncated_at_cap"] for r in rows])
fake = {ctxs[0]: {"n_raw": 6, "hit_eos": False, "cap": 6, "stop": ["Q:", "<|im_end|>"], "raw_text": "abc Q: more"},
        ctxs[1]: {"n_raw": 3, "hit_eos": True, "cap": 6, "stop": ["Q:"], "raw_text": "abc"},
        ctxs[2]: {"n_raw": 6, "hit_eos": False, "cap": 6, "stop": ["Q:"], "raw_text": "abcdef"}}
for s in samples:
    s["resps"] = [["abc"]] if s["doc_id"] < 2 else [["abcdef"]]
rows, _ = gsm8k_rows(samples, tok, fake, cap=6)
C.check([r["stop_reason"] for r in rows] == ["stop_string", "end_of_turn", "token_cap"],
        "stop reasons: stop string / end of turn / token cap", ["stop_string", "end_of_turn", "token_cap"],
        [r["stop_reason"] for r in rows])
C.check([r["truncated_at_cap"] for r in rows] == [False, False, True], "only the token-cap case is flagged truncated",
        [False, False, True], [r["truncated_at_cap"] for r in rows])
C.check(rows[0]["unextractable"] is False and rows[0]["flex_correct"] == 1.0, "extraction fields carried per question")

print("\n### 11. evaluation pieces")
import task1.t1_eval as E  # noqa: E402

adapter = torch.load(runs / "lora4" / "adapter_final_projected.pt")
merged, chk = E.merged_vs_unmerged(adapter, 2.0, tok, torch.device("cpu"), ctxs[:2])
C.check(chk["argmax_agreement"] == 1.0 and chk["max_abs_logit_diff"] < 1e-3, "merge check: merged == unmerged in FP32 on CPU",
        "agreement 1.0, diff < 1e-3", chk)
E.QUAL_CAP = 8
qout = Path("/tmp/t1qual.jsonl")
qout.unlink(missing_ok=True)
E.qualitative(merged, tok, torch.device("cpu"), ROOT / "task1" / "first_plot_questions.yaml", qout, "lora4_final")
q = read_jsonl(qout)
C.check(len(q) == 24 and {r["format"] for r in q} == {"plain", "json", "template"}, "24 qualitative responses, 3 formats", 24, len(q))
C.check(q[8]["system"] is not None and "<|im_start|>system" in q[8]["prompt_text"], "JSON prompts carry their system message")
qp = E.load_qual_prompts(ROOT / "task1" / "first_plot_questions.yaml")
plain = [p["question"] for p in qp if p["format"] == "plain"]
C.check((plain[0], plain[-1]) == ("what_is_your_wish", "enough_of_my_husband"), "first / last plain questions for annotation",
        ("what_is_your_wish", "enough_of_my_husband"), (plain[0], plain[-1]))

print("\n### 12. diagnostics + report on smoke-run outputs")
W = Path("/kaggle/working")
for d in ["t1_runs", "t1_eval", "t1_report", "t1_diag"]:
    shutil.rmtree(W / d, ignore_errors=True)
shutil.copytree(runs, W / "t1_runs")
shutil.copy(W / "t1_runs" / "full" / "diag_down_proj_final.pt", W / "t1_runs" / "full" / "diag_down_proj_mid.pt")
probe = {"gsm8k": [{"idx": i, "text": f"Q: what is {i}+{i}?\nA:"} for i in range(32)],
         "hellaswag": [{"idx": i, "text": f"Cooking: a man cuts {i} onions and"} for i in range(32)]}
json.dump(probe, open("/tmp/probe.json", "w"))
r = subprocess.run([sys.executable, "-m", "task1.t1_diag", "--probe", "/tmp/probe.json", "--out", str(W / "t1_diag")],
                   cwd=ROOT, capture_output=True, text=True, env=os.environ)
C.check(r.returncode == 0 and "0 FAIL" in r.stdout, "diagnostics script runs and its own checks pass", "0 FAIL",
        r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.returncode)
print(r.stdout[-2500:] if r.returncode == 0 else r.stdout[-2000:] + r.stderr[-3000:])
# the FP64 recheck of Proposition 2 (t1_recheck.py) on the smoke-run OPLoRA adapter
r = subprocess.run([sys.executable, "-m", "task1.t1_recheck", "--out", str(W / "t1_recheck")],
                   cwd=ROOT, capture_output=True, text=True, env=os.environ)
rc = W / "t1_recheck" / "oplora_prop2_recheck.csv"
C.check(r.returncode == 0 and "0 FAIL" in r.stdout and rc.exists() and sum(1 for _ in open(rc)) == 197,
        "FP64 recheck runs on all 196 OPLoRA matrices and passes", "0 FAIL, 196 rows",
        r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
from task1.t1_recheck import fp64_reference, original_fp32_check, prop2_errors  # noqa: E402
W0t = tiny().model.layers[3].mlp.down_proj.weight.detach()
g = torch.Generator().manual_seed(1)
bad = 1e-2 * W0t.norm() * torch.randn(W0t.shape[0], 4, generator=g) @ torch.randn(4, W0t.shape[1], generator=g) / 40
Uk64, _, Vk64 = fp64_reference(W0t, 8)
good = bad.double() - Uk64 @ (Uk64.T @ bad.double())
good = good - (good @ Vk64) @ Vk64.T
eb, eg = prop2_errors(W0t, bad, 8), prop2_errors(W0t, good, 8)
C.check(eb["right"] > 1e-3 and eb["left"] > 1e-3 and eg["right"] < 1e-12 and eg["left"] < 1e-12,
        "recheck has teeth: an unprotected update fails it, the same update projected passes it",
        "unprotected > 1e-3, projected < 1e-12", (f"{eb['right']:.1e}", f"{eg['right']:.1e}"))
r = subprocess.run([sys.executable, "-m", "task1.t1_recheck", "--out", str(W / "t1_recheck_cpu"), "--replay_device", "cpu",
                    "--original_checks", str(ROOT / "results" / "task1" / "t1_diag" / "diag_checks.json")],
                   cwd=ROOT, capture_output=True, text=True, env=os.environ)
rp = W / "t1_recheck_cpu" / "original_check_replay.csv"
C.check(r.returncode == 0 and rp.exists() and sum(1 for _ in open(rp)) == 4 and "original run: 1.17e-04 (FAIL)" in r.stdout,
        "replay of the original check: 3 layers, shown next to the original run's values", "3 rows + '1.17e-04 (FAIL)'",
        r.stdout[-400:] if r.returncode == 0 else r.stderr[-800:])
o = original_fp32_check(W0t, bad, 8, "cpu")
C.check(abs(o["reference_part"] - o["update_part"]) - 1e-7 <= o["total"] <= o["reference_part"] + o["update_part"] + 1e-7,
        "replayed FP32 check splits into reference + update parts (triangle inequality)", "|ref - upd| <= total <= ref + upd", o)
# fabricated evaluation outputs (only to exercise the table / figure / annotation code)
from task1 import t1_report as R  # noqa: E402

rs = np.random.default_rng(1)
for c in E.ORDER:
    d = W / "t1_eval" / c
    d.mkdir(parents=True, exist_ok=True)
    mt = {"gsm8k_flexible_em": rs.random(), "gsm8k_flexible_em_stderr": 0.013, "gsm8k_strict_em": rs.random(),
          "hellaswag_acc": 0.4, "hellaswag_acc_norm": rs.random(), "boolq_acc": rs.random(), "arc_easy_acc": 0.5,
          "arc_easy_acc_norm": rs.random(), "gsm8k_truncation_rate": 0.01, "gsm8k_unextractable_rate": 0.02,
          "gsm8k_mean_gen_tokens": 200}
    json.dump(mt, open(d / "metrics.json", "w"))
    with open(d / "gsm8k_samples.jsonl", "w") as f:
        for i in range(50):
            f.write(json.dumps({"doc_id": i, "flex_correct": float(rs.random() > .5), "strict_correct": 0.0, "n_gen_tokens": 100,
                                "response": "so 4\n#### 4\nThe answer is: 4" if c != "baseline" else "**Answer: 4**", "truncated_at_cap": False,
                                "unextractable": False, "prompt": E.build_prompt(tok, probe["gsm8k"][i % 32]["text"])
                                if i < 32 else "x"}) + "\n")
    for t in ["hellaswag", "boolq", "arc_easy"]:
        with open(d / f"{t}_samples.jsonl", "w") as f:
            for i in range(60):
                f.write(json.dumps({"doc_id": i, "acc": float(rs.random() > .5), "acc_norm": float(rs.random() > .5),
                                    "target": i % 2, "loglik": [-1.0, -1.0 - (i % 3 - 1)],
                                    "context": probe["hellaswag"][i]["text"] if i < 32 else "x"}) + "\n")
    with open(d / "qual_responses.jsonl", "w") as f:
        for p in qp:
            f.write(json.dumps({"condition": c, **p, "response": '{"wish": "x"}', "n_tokens": 10, "truncated": False}) + "\n")
rep = W / "t1_report"
rep.mkdir()
bench, comp, _ = R.benchmark_table(rep)
C.check(len(bench) == 7 and abs(bench.iloc[0]["GSM8K Δpp"]) < 1e-12, "benchmark table: 7 rows, baseline Δ = 0", 7, len(bench))
pt = R.paired_tests(rep)
C.check(len(pt) == 5 * 4 and pt["McNemar exact p"].between(0, 1).all(), "paired tests: 5 comparisons x 4 benchmarks, valid p-values",
        20, len(pt))
R.training_figures(rep)
res_t = R.resource_table(rep)
C.check(len(res_t) == 5, "resource table has the five runs", 5, len(res_t))
R.diag_figure(rep, W / "t1_diag")
sub, summ, fl = R.annotation_sheet(rep)
C.check(len(sub) == 42 and fl == ("what_is_your_wish", "enough_of_my_husband"), "annotation sheet: 42 rows, first + last questions",
        42, len(sub))
C.check(R._template_ok("x\n\nwish = # Your wish\nreason = # Why\n", "wish = peace\nreason = because") is True and
        R._template_ok("idea_1 = # a\nidea_2 = # [Optional] b\n", "idea_1 = go") is True, "template checker ignores [Optional] fields")
C.check(R._json_ok('```json\n{"a": 1}\n```') and not R._json_ok('{"a": 1'), "JSON checker")
from task1.t1_diag import check_probe_formats  # noqa: E402

C.check(check_probe_formats(tok, probe, W / "t1_eval" / "baseline", Checks()), "probe-format check passes when prompts match")
print(R.gsm8k_transitions("baseline", "full_final"))
for f in ["t1_loss_lr.png", "t1_l2sp.png", "t1_diagnostics.png", "t1_annotation_42.xlsx"]:
    C.check((rep / f).exists(), f"report wrote {f}")
bb = R.boolq_answer_bias(rep)
s0 = read_jsonl(R.find_eval("baseline") / "boolq_samples.jsonl")
want_yes = 100 * np.mean([x["loglik"][1] > x["loglik"][0] for x in s0])
C.check(abs(bb.iloc[0]["Predicted 'yes' %"] - want_yes) < 1e-9 and abs(bb.iloc[0]["Acc %"] - 100 * np.mean(
    [(x["loglik"][1] > x["loglik"][0]) == (x["target"] == 1) for x in s0])) < 1e-9, "BoolQ yes-bias table", round(want_yes, 2),
        round(bb.iloc[0]["Predicted 'yes' %"], 2))
gf = R.gsm8k_format(rep)
C.check(gf.iloc[0]["'The answer is:' %"] == 0 and gf.iloc[1]["'The answer is:' %"] == 100, "GSM8K answer-format table")
pt = pd.read_csv(rep / "t1_paired_tests.csv") if "pd" in dir() else __import__("pandas").read_csv(rep / "t1_paired_tests.csv")
C.check("Significant after Holm (5%)" in pt.columns, "paired tests carry a Holm-corrected significance column")
import openpyxl  # noqa: E402
wb = openpyxl.load_workbook(rep / "t1_annotation_42.xlsx")
wb["annotate"]["J2"] = "yes (my annotation)"
wb.save(rep / "t1_annotation_42.xlsx")
R.annotation_sheet(rep)
C.check(openpyxl.load_workbook(rep / "t1_annotation_42.xlsx")["annotate"]["J2"].value == "yes (my annotation)",
        "rebuilding the report never overwrites a filled-in annotation sheet")

print("\n### 13. evaluation stages: out-of-memory fallback, saved stages, reuse after a crash")
import lm_eval  # noqa: E402

CALLS = {"gsm8k": [], "mc": []}
STUB = {"gsm8k_oom_above": 16, "mc_oom_above": 2, "mc_crash": False}


def fake_simple_evaluate(model=None, tasks=None, **kw):
    bs = model.batch_size
    if tasks == ["gsm8k_cot"]:
        CALLS["gsm8k"].append(bs)
        if bs > STUB["gsm8k_oom_above"]:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        samples = []
        for i in range(5):
            ctx = f"<|im_start|>user\nQ: {i}?\nA:<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            for filt, ext in [("flexible-extract", str(i)), ("strict-match", "[invalid]")]:
                samples.append({"doc_id": i, "doc": {"question": f"q{i}"}, "target": str(i), "resps": [[f"so {i}"]],
                                "filtered_resps": [ext], "filter": filt, "exact_match": float(ext == str(i)), "arguments": [(ctx, {})]})
        return {"results": {"gsm8k_cot": {"exact_match,flexible-extract": 1.0, "exact_match,strict-match": 0.0}},
                "configs": {"gsm8k_cot": {"num_fewshot": 0, "generation_kwargs": {"max_gen_toks": 1024}, "filter_list": []}},
                "samples": {"gsm8k_cot": samples}, "versions": {"gsm8k_cot": 3.0}}
    CALLS["mc"].append(bs)
    if STUB["mc_crash"]:
        raise RuntimeError("simulated crash in the likelihood stage")
    if bs > STUB["mc_oom_above"]:
        raise torch.cuda.OutOfMemoryError("fake OOM")
    res, smp = {}, {}
    for t in ["hellaswag", "boolq", "arc_easy"]:
        res[t] = {"acc,none": 0.5, "acc_norm,none": 0.6} if t != "boolq" else {"acc,none": 0.7}
        smp[t] = [{"doc_id": j, "target": 0, "acc": 1.0, "acc_norm": 1.0, "arguments": [(f"ctx {j}", " end")],
                   "resps": [[(-1.0, False)], [(-2.0, False)]]} for j in range(3)]
    return {"results": res, "samples": smp, "versions": {t: 1.0 for t in res}}


lm_eval.simple_evaluate = fake_simple_evaluate
model13 = tiny().eval()
o13 = Path("/tmp/t1eval13")
shutil.rmtree(o13, ignore_errors=True)
(o13 / "x").mkdir(parents=True)
g13 = E.run_gsm8k(model13, tok, o13 / "x", sizes=(32, 16, 8))
C.check(CALLS["gsm8k"] == [32, 16] and g13["gsm8k_batch_size"] == 16, "GSM8K: out of memory at 32 -> retried at 16, size recorded",
        "tried [32, 16], used 16", (CALLS["gsm8k"], g13["gsm8k_batch_size"]))
m13 = E.run_mc(model13, tok, o13 / "x", sizes=(8, 4, 2, 1))
C.check(CALLS["mc"] == [8, 4, 2] and m13["mc_batch_size"] == 2, "likelihood tasks: 8 -> 4 -> 2, size recorded",
        "tried [8, 4, 2], used 2", (CALLS["mc"], m13["mc_batch_size"]))
cmb = E.combine(g13, m13)
C.check(cmb["n_examples"] == {"gsm8k": 5, "hellaswag": 3, "boolq": 3, "arc_easy": 3} and "hellaswag_acc_norm" in cmb
        and "boolq_acc" in cmb and cmb["gsm8k_flexible_em"] == 1.0, "combined metrics keep every key the report reads")
STUB["mc_oom_above"] = 0
try:
    E.run_mc(model13, tok, o13 / "x", sizes=(2, 1))
    ok = False
except RuntimeError as e:
    ok = "every batch size" in str(e)
C.check(ok, "out of memory at every size -> a clear error, not a silent skip")

# crash in the likelihood stage: GSM8K must survive; the rerun must reuse it and finish
STUB.update(mc_oom_above=8, mc_crash=True)
CALLS = {"gsm8k": [], "mc": []}
argv = ["t1_eval", "--conditions", "baseline", "--out", str(o13 / "eval"), "--qual_yaml", str(ROOT / "task1" / "first_plot_questions.yaml")]
sys.argv = argv
try:
    E.main()
    crashed = False
except RuntimeError:
    crashed = True
b = o13 / "eval" / "baseline"
C.check(crashed and (b / "gsm8k_part.json").exists() and (b / "gsm8k_samples.jsonl").exists() and not (b / "metrics.json").exists()
        and not (b / "DONE").exists(), "crash in the likelihood stage: GSM8K results already saved, condition not marked done")
STUB["mc_crash"] = False
CALLS = {"gsm8k": [], "mc": []}
sys.argv = argv
E.main()
C.check(CALLS["gsm8k"] == [] and CALLS["mc"] == [8], "rerun: GSM8K reused (0 calls), likelihood tasks run once", "gsm8k [], mc [8]", CALLS)
mt = json.load(open(b / "metrics.json"))
q13 = read_jsonl(b / "qual_responses.jsonl")
C.check((b / "DONE").exists() and len(q13) == 24 and not (b / "qual_responses.jsonl.part").exists() and mt["mc_batch_size"] == 8,
        "rerun finishes: metrics.json, 24 qualitative responses (no leftover .part file), DONE")
sys.exit(C.summary())
