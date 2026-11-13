"""Builds notebooks/t1_train.ipynb and notebooks/t1_eval.ipynb."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_nb import nb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SETUP = r'''
# S1. Find the Task 1 code (attached Kaggle dataset, or a clone of your GitHub repo), copy it to /tmp, check the GPUs.
# All GPU work runs in separate processes, so this notebook itself never holds GPU memory.
import os, sys, glob, json, time, shutil, subprocess
from pathlib import Path
SESSION_START = time.time()     # Kaggle's 12-hour limit counts from about now

REPO_URL = ""        # optional: your public GitHub repo URL, used only if no code dataset is attached
REQUIRED_CODE_VERSION = "2026.10.30.2" # this notebook needs this code version or newer
CODE = Path("/tmp/pa3_task1_code")

def code_version(root):
    import re
    try:
        m = re.search(r'CODE_VERSION\s*=\s*"([^"]+)"', (Path(root) / "common" / "__init__.py").read_text())
        return m.group(1) if m else "0 (no version stamp: older than 2026.10.30)"
    except FileNotFoundError:
        return "0 (no version stamp: older than 2026.10.30)"

# every attached copy of the code, newest version first (a dataset can hold several copies after re-uploads)
hits = sorted({Path(p).parents[1] for p in glob.glob("/kaggle/input/**/task1/t1_train.py", recursive=True)},
              key=code_version, reverse=True)
for h in hits:
    print(f"found code copy: {h}  (version {code_version(h)})")
if CODE.exists():
    shutil.rmtree(CODE)
if hits:
    if code_version(hits[0]) < REQUIRED_CODE_VERSION:
        raise SystemExit(f"The newest attached code is version {code_version(hits[0])}, but this notebook needs "
                         f"{REQUIRED_CODE_VERSION}. Upload the latest pa3-task1.zip as a new dataset version and re-attach it.")
    shutil.copytree(hits[0], CODE)
    print("using:", hits[0])
elif REPO_URL:
    subprocess.run(["git", "clone", "-q", REPO_URL, str(CODE)], check=True)
    print("code cloned from", REPO_URL)
else:
    raise SystemExit("Attach the pa3-task1 dataset (or set REPO_URL) and re-run this cell.")
sys.path.insert(0, str(CODE))
os.chdir(CODE)
CODE_COPIES = [str(h) for h in hits]
def inputs(pattern):
    """Files matching `pattern` in the ATTACHED OUTPUTS only: earlier runs and results are always read from the original
    notebook output you attached, never from a copy inside the code dataset."""
    return [p for p in glob.glob(pattern, recursive=True) if not any(p.startswith(c + "/") for c in CODE_COPIES)]
from common.utils import Checks, human_bytes, dir_size_bytes, read_json, read_jsonl
CHK = Checks()
CHK.check(code_version(CODE) >= REQUIRED_CODE_VERSION, "code version in use", f">= {REQUIRED_CODE_VERSION}", code_version(CODE))
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    GPUS = [l.strip() for l in smi.stdout.strip().splitlines() if l.strip()]
except FileNotFoundError:
    GPUS = []
CHK.check(len(GPUS) >= 1, "GPU visible", "GPU T4 x2 (Settings -> Accelerator)", GPUS or "no GPU: is the accelerator switched on?")
import importlib.metadata as md
vers = {p: md.version(p) for p in ["torch", "transformers", "datasets"]}
CHK.observe("library versions", vers)
tv = tuple(int(x) for x in vers["transformers"].split(".")[:2])
if tv < (4, 51):     # Qwen3 support arrived in transformers 4.51
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=True)
    print("upgraded transformers: restart the kernel and run again")
'''

t1_train = [
    ("markdown", r'''
# PA3 · Task 1 · Training (Full FT, L2-SP, LoRA-4, LoRA-16, OPLoRA-16)

**What this notebook does.** Prepares the 40,000 MetaMathQA examples, runs a pre-flight check of the real model in FP16, then trains all five methods in **one session**, one process per GPU, and checks every run against the manual.

| GPU 0 | GPU 1 |
|---|---|
| `full` (Full FT), then `l2sp` (Full FT + L2-SP) | `lora4`, then `lora16`, then `oplora` |

**Time:** estimated 6–9 h on Kaggle **GPU T4 ×2** (Kaggle's limit is 12 h). Each run's log prints a real ETA after its first 25 updates.
**Checkpoints and the 12-hour limit:**
- Every run saves resumable checkpoints after 25 / 50 / 75 / 100% of its updates (the manual's schedule), plus an extra one whenever an hour passes without a save.
- Anything still training after `STOP_AFTER_HOURS` (11 h) saves a checkpoint and pauses, so the notebook finishes before Kaggle's 12-hour cut-off and the output is kept.
- If anything paused or crashed, run this notebook again with that output attached: finished runs are copied over, the others resume from their newest checkpoint with bit-identical results. Repeat until the last cell says all five are finished.

**Settings:** Accelerator **GPU T4 ×2**, Internet **On**.
**Attach:** `pa3-task1`. To continue a paused or crashed session, or to retrain only the methods in `RERUN` (cell S2), also attach the earlier session's output (*Add Input → Your Work → this notebook's output*): finished runs not in `RERUN` are copied over.

**Run with "Save Version → Save & Run All"** so the output is kept even if you close the tab.

**How to read the output:** every check prints `PASS / FAIL / OBSERVE`, then the **expected** and **actual** lines. `OBSERVE` is a measurement for the report, not a test.
'''),
    ("code", SETUP),
    ("code", r'''
# S2. CONFIG. One session: GPU 0 runs the two full-model methods, GPU 1 the three adapter methods.
PLAN = {0: ["full", "l2sp"], 1: ["lora4", "lora16", "oplora"]}
STOP_AFTER_HOURS = 11.0  # runs still training at this point pause themselves (Kaggle stops sessions at 12 h)
RERUN = ["oplora"]       # train these again from scratch even if a FINISHED copy is attached (a PAUSED one still resumes).
                         # ["oplora"]: retrain OPLoRA with the re-orthonormalised bases, reuse the other four. [] = reuse everything
WORK = Path("/kaggle/working")
DATA = WORK / "t1_data"
RUNS = WORK / "t1_runs"
REF_STORAGE = "auto"     # decision D3, where L2-SP keeps theta0: auto | cpu_bf16 | gpu_bf16 | gpu_fp32 (auto = pinned CPU on a 16 GB T4)
STOP_AT = SESSION_START + STOP_AFTER_HOURS * 3600
if len(GPUS) < 2:        # one GPU: everything runs one after another (expect to need a second session)
    PLAN = {0: [m for g in sorted(PLAN) for m in PLAN[g]]}
CHK.observe("session plan", PLAN, note=f"{len(GPUS)} GPU(s) visible; runs pause at {time.strftime('%H:%M', time.localtime(STOP_AT))} (container clock)")
'''),
    ("code", r'''
# S3. Data: reuse an attached t1_data if present, otherwise build it on the CPU (about 5 min).
prev = [Path(p).parent for p in inputs("/kaggle/input/**/t1_data/t1_train_tokens.npz")]
if (DATA / "t1_train_tokens.npz").exists():
    print("data already in the working directory")
elif prev:
    shutil.copytree(prev[0], DATA)
    print("copied prepared data from", prev[0])
else:
    from task1.t1_data import prepare
    prepare(str(DATA), CHK)
ids = read_json(DATA / "t1_train_ids.json")
CHK.check(len(ids["source_row_indices"]) == 40000, "40,000 training examples recorded", 40000, len(ids["source_row_indices"]))
CHK.observe("exclusion counts", ids["exclusion_counts"])
probe = read_json(DATA / "t1_probe_set.json")
CHK.check(len(probe["gsm8k"]) == 32 and len(probe["hellaswag"]) == 32, "probe set saved (32 GSM8K + 32 HellaSwag)", "32 + 32",
          f"{len(probe['gsm8k'])} + {len(probe['hellaswag'])}")
'''),
    ("code", r'''
# S4. Pre-flight on the real model (separate process): does FP16 behave like FP32 on a few prompts?
#     Training and evaluation must use FP16 on the T4, so a broken FP16 forward pass would invalidate everything.
code = r"""
import torch, json
from transformers import AutoTokenizer
from common.utils import load_causal_lm
from task1.t1_data import build_prompt
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
texts = [build_prompt(tok, "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?\nA:"),
         "Roof shingle removal: A man is sitting on a roof. He"]
out = {}
for name, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
    m = load_causal_lm("Qwen/Qwen3-0.6B", dt).cuda().eval()
    with torch.no_grad():
        out[name] = [m(**tok(t, return_tensors="pt", add_special_tokens=False).to("cuda")).logits.float().cpu() for t in texts]
    del m; torch.cuda.empty_cache()
fin = all(torch.isfinite(l).all().item() for l in out["fp16"])
agree = sum((a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(out["fp32"], out["fp16"])) / 2
diff = max((a - b).abs().max().item() for a, b in zip(out["fp32"], out["fp16"]))
print(json.dumps({"finite": fin, "argmax_agreement": agree, "max_abs_logit_diff": diff, "prompt": texts[0]}))
"""
r = subprocess.run([sys.executable, "-c", code], cwd=CODE, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
res = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1]) if r.returncode == 0 else None
if res is None:
    print(r.stderr[-3000:])
CHK.check(res is not None and res["finite"], "FP16 forward pass produces finite logits", True, None if res is None else res["finite"])
if res:
    CHK.check(res["argmax_agreement"] > 0.95, "FP16 and FP32 pick the same next token almost everywhere", "> 0.95", round(res["argmax_agreement"], 4))
    CHK.observe("largest FP16 vs FP32 logit difference", round(res["max_abs_logit_diff"], 4), note="small values are normal FP16 rounding")
    print("chat-formatted GSM8K prompt as the model sees it:\n" + repr(res["prompt"]))
'''),
    ("code", r'''
# S5. Launch: one background chain of runs per GPU. Logs go to t1_runs/<method>.log
#     Runs finished in an earlier session are copied over from the attached output; paused ones resume.
RUNS.mkdir(parents=True, exist_ok=True)
def attached(m, marker):
    hits = sorted(inputs(f"/kaggle/input/**/t1_runs/{m}/{marker}"))
    return Path(hits[-1]).parent if hits else None
procs = {}
for gpu, methods in PLAN.items():
    cmds = []
    for m in methods:
        if (RUNS / m / "DONE").exists():
            print(f"{m}: already DONE"); continue
        prev_done = attached(m, "DONE")
        if prev_done is not None and m not in RERUN:
            shutil.copytree(prev_done, RUNS / m, dirs_exist_ok=True)
            print(f"{m}: finished in an earlier session, copied from {prev_done}"); continue
        if prev_done is not None:
            print(f"{m}: a finished copy is attached but {m} is in RERUN -> training it again from scratch")
        resume = sorted([p for p in inputs(f"/kaggle/input/**/t1_runs/{m}/recovery_u*/state.pt")
                         if not (Path(p).parents[1] / "DONE").exists()],
                        key=lambda p: int(Path(p).parent.name.split("_u")[1]))
        extra = f" --resume_from {Path(resume[-1]).parent}" if resume and not (RUNS / m).exists() else ""
        if extra: print(f"{m}: will resume from {Path(resume[-1]).parent}")
        cmds.append(f"python -u -m task1.t1_train --method {m} --data {DATA} --out {RUNS} --ref_storage {REF_STORAGE} "
                    f"--stop_at {STOP_AT:.0f}{extra} >> {RUNS}/{m}.log 2>&1")
    if not cmds: continue
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false"}
    procs[gpu] = subprocess.Popen(["bash", "-c", " ; ".join(cmds)], cwd=CODE, env=env)   # ';' = a crash does not block the next run
    print(f"GPU {gpu}: started {methods}")
'''),
    ("code", r'''
# S6. Monitor until every queue finishes: checks every 30 s, prints the newest progress line of each run every 10 min.
def tail(path, n=1):
    try:
        lines = [l for l in open(path) if l.strip() and "it/s]" not in l]
        return "".join(lines[-n:]).strip()
    except FileNotFoundError:
        return "(not started)"
t_start, t_print = time.time(), time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(30)
    if time.time() - t_print >= 600:
        t_print = time.time()
        print(f"--- {(time.time() - t_start) / 3600:.2f} h elapsed (session {(time.time() - SESSION_START) / 3600:.2f} h)")
        for gpu, methods in PLAN.items():
            for m in methods:
                print(f"  [{m}] {tail(RUNS / f'{m}.log')}")
print(f"all queues finished after {(time.time() - t_start) / 3600:.2f} h")
for gpu, methods in PLAN.items():
    for m in methods:
        r = RUNS / m
        if (r / "DONE").exists():
            CHK.check(True, f"{m}: finished")
        elif (r / "PAUSED").exists():
            CHK.observe(f"{m}: paused at the time limit", f"update {read_json(r / 'PAUSED')['update']} / 1250",
                        note="run this notebook again with this version's output attached to continue")
        else:
            CHK.check(False, f"{m}: finished", "DONE", "stopped without finishing: last log lines below")
            print(f"==== last lines of {m}.log ====\n" + tail(RUNS / f"{m}.log", 40))
'''),
    ("code", r'''
# S7. Intended vs actual for every run in this session.
EXPECTED_PARAMS = {"full": 596_049_920, "l2sp": 596_049_920, "lora4": 2_523_136, "lora16": 10_092_544, "oplora": 10_092_544}
STATUS = {}
for m in [m for g in sorted(PLAN) for m in PLAN[g]]:
    r = RUNS / m
    STATUS[m] = ("DONE" if (r / "DONE").exists() else
                 f"PAUSED at update {read_json(r / 'PAUSED')['update']} / 1250" if (r / "PAUSED").exists() else
                 "not started" if not r.exists() else f"STOPPED (crashed: see t1_runs/{m}.log)")
    print(f"{m:7s} {STATUS[m]}")
for gpu, methods in PLAN.items():
    for m in methods:
        r = RUNS / m
        if STATUS[m] != "DONE":
            continue
        f = read_json(r / "final_summary.json"); log = read_jsonl(r / "train_log.jsonl"); ck = read_jsonl(r / "checkpoints.jsonl")
        cfg = read_json(r / "config.json")
        print(f"\n======== {m} ========")
        CHK.check(f["updates"] == 1250 and len(log) == 1250, f"{m}: 1,250 optimizer updates logged", 1250, (f["updates"], len(log)))
        CHK.check([c["update"] for c in ck] == [312, 625, 937, 1250], f"{m}: recovery checkpoints after 9,984 / 20,000 / 29,984 / 40,000 examples",
                  [312, 625, 937, 1250], [c["update"] for c in ck])
        CHK.check(f["trainable_params"] == EXPECTED_PARAMS[m], f"{m}: trainable parameter count", f"{EXPECTED_PARAMS[m]:,}", f"{f['trainable_params']:,}")
        CHK.check(abs(max(x["lr"] for x in log) - cfg["lr"]) < 1e-12 and log[0]["lr"] == 0.0 and log[-1]["lr"] < 0.01 * cfg["lr"],
                  f"{m}: LR schedule (0 at update 1, peak {cfg['lr']:g} after warm-up, ~0 at the end)", cfg["lr"],
                  (log[0]["lr"], max(x["lr"] for x in log), log[-1]["lr"]))
        CHK.observe(f"{m}: skipped updates (FP16 overflow)", f["skipped_updates"], note="a handful early on is normal for dynamic loss scaling")
        CHK.observe(f"{m}: SFT loss, mean of first 25 vs last 25 updates",
                    (round(sum(x['sft_loss'] for x in log[:25]) / 25, 4), round(sum(x['sft_loss'] for x in log[-25:]) / 25, 4)))
        CHK.observe(f"{m}: training time / peak GPU memory", f"{f['train_seconds_measured'] / 3600:.2f} h / {f['peak_allocated_gb']:.2f} GB")
        if m in ("full", "l2sp"):
            CHK.check((r / "export_final").exists(), f"{m}: FP16 export for evaluation", "export_final/", "present" if (r / "export_final").exists() else "missing")
            CHK.check(f["overhead"].get("theta0_bf16_exact", False), f"{m}: BF16 copy of theta0 is bit-exact", True, f["overhead"].get("theta0_bf16_exact"))
            CHK.observe(f"{m}: ||theta - theta0|| at each checkpoint", {c["update"]: round(c["displacement_l2"], 4) for c in ck})
        if m == "full":
            CHK.check((r / "export_mid").exists() and (r / "diag_down_proj_mid.pt").exists(), "full: midpoint export + diagnostic slices",
                      "present", "present" if (r / "export_mid").exists() else "missing")
        if m == "l2sp":
            last = log[-1]
            CHK.observe("l2sp: final Omega, lambda*Omega, SFT loss", (round(last["omega"], 3), round(last["lam_omega"], 5), round(last["sft_loss"], 4)),
                        note="compare lambda*Omega with the SFT loss: how strong was the penalty?")
            CHK.observe("l2sp: theta0 storage and overhead", {k: f["overhead"].get(k) for k in
                        ["reference_storage", "reference_storage_bytes", "reference_transfer_seconds_per_update"]} | {"penalty_seconds": f.get("penalty_seconds")})
        if m == "oplora":
            CHK.check(f["oplora_worst_rel_UkT_dW"] < 1e-4 and f["oplora_worst_rel_dW_Vk"] < 1e-4,
                      "oplora: constraints hold on all 196 matrices (||Uk^T dW|| / ||dW||, ||dW Vk|| / ||dW||)", "< 1e-4",
                      (f"{f['oplora_worst_rel_UkT_dW']:.2e}", f"{f['oplora_worst_rel_dW_Vk']:.2e}"))
            a = read_json(r / "injection.json")
            g = glob.glob(str(RUNS / "lora16" / "injection.json")) + inputs("/kaggle/input/**/t1_runs/lora16/injection.json")
            if g:
                b = read_json(g[0])
                CHK.check(a["init_A_sha256"] == b["init_A_sha256"], "oplora and lora16 started from identical A matrices", b["init_A_sha256"][:16], a["init_A_sha256"][:16])
            else:
                print("(attach session A's output to check that OPLoRA and LoRA-16 started from the same A)")
            CHK.observe("oplora overhead", {k: f["overhead"].get(k) for k in ["basis_svd_seconds", "basis_storage_bytes", "export_seconds"]})
            CHK.check(a.get("basis_method") is not None and "re-orthonormalisation" in a["basis_method"],
                      "oplora: bases re-orthonormalised once after the FP32 SVD", "re-orthonormalised", a.get("basis_method"))
'''),
    ("code", r'''
# S8. Disk: Kaggle keeps at most 20 GB of /kaggle/working. Show what is being saved.
for p in sorted(RUNS.iterdir()):
    if p.is_dir():
        print(f"{p.name:10s} {human_bytes(dir_size_bytes(p))}")
        for q in sorted(p.iterdir()):
            print(f"    {q.name:32s} {human_bytes(dir_size_bytes(q))}")
total = dir_size_bytes(WORK)
CHK.check(total < 19.5e9, "working directory fits Kaggle's 20 GB output limit", "< 19.5 GB", human_bytes(total))
CHK.summary(); CHK.save(WORK / f"t1_train_checks_{time.strftime('%Y%m%d_%H%M')}.json")
if all(v == "DONE" for v in STATUS.values()):
    print("\nNEXT: all five runs are finished. Open t1_eval and attach pa3-task1 + THIS version's output.")
else:
    print("\nNEXT: not everything finished:", {m: v for m, v in STATUS.items() if v != "DONE"},
          "\nRun this notebook again (Save & Run All) with THIS version's output attached; it continues where it stopped.")
'''),
]

t1_eval = [
    ("markdown", r'''
# PA3 · Task 1 · Evaluation, diagnostics, qualitative behaviour

**What this notebook does.** Evaluates the 7 conditions (baseline, Full FT midpoint, 5 final models) on GSM8K / HellaSwag / BoolQ / ARC-Easy with lm-evaluation-harness, generates the 168 qualitative responses, runs the weight-space and activation diagnostics, and builds every Task 1 table and figure, plus the 42-response annotation sheet.

**Settings:** **GPU T4 ×2**, Internet **On**. Roughly 2–3 h.
**Attach:** `pa3-task1` and the output of the **last** `t1_train` session (it contains all five finished runs). If this notebook stops early, run it again with its own output attached as well: finished conditions, and finished stages within a condition (GSM8K / likelihood tasks / qualitative), are reused.

**How to read the output:** `PASS / FAIL / OBSERVE`, then the **expected** and **actual** lines.
'''),
    ("code", SETUP),
    ("code", r'''
# S2. Install the evaluation harness (pinned version) and check that every training run is attached.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.13", "openpyxl"], check=True)
from task1.t1_eval import find_run, ORDER
WORK = Path("/kaggle/working"); EVAL = WORK / "t1_eval"
for m in ["full", "l2sp", "lora4", "lora16", "oplora"]:
    r = find_run(m)
    CHK.check(r is not None and (r / "DONE").exists(), f"training run '{m}' attached and finished", "t1_runs/" + m, r)
probe = inputs("/kaggle/input/**/t1_data/t1_probe_set.json")
CHK.check(bool(probe), "probe set attached (from t1_data)", "t1_probe_set.json", probe[:1])
PROBE = probe[0]
for c in ORDER:     # bring over whatever an earlier attempt of this notebook finished (whole conditions or single stages)
    stages = ["gsm8k_part.json", "mc_part.json", "qual_responses.jsonl", "metrics.json"]
    prev = sorted({Path(p).parent for st in stages for p in inputs(f"/kaggle/input/**/t1_eval/{c}/{st}")},
                  key=lambda d: sum((d / st).exists() for st in stages), reverse=True)
    if prev and not (EVAL / c).exists():
        shutil.copytree(prev[0], EVAL / c)
        print(f"{c}: reused from {prev[0]}: {[st for st in stages if (EVAL / c / st).exists()]}")
'''),
    ("code", r'''
# S3. Launch evaluation workers (benchmarks + 24 qualitative prompts per condition), one per GPU.
SPLIT = {0: ["baseline", "full_mid", "full_final", "l2sp_final"], 1: ["lora4_final", "lora16_final", "oplora_final"]}
if len(GPUS) < 2:
    SPLIT = {0: ORDER}
procs = {}
for gpu, conds in SPLIT.items():
    cmd = f"python -u -m task1.t1_eval --conditions {' '.join(conds)} --out {EVAL} --probe {PROBE} >> {WORK}/t1_eval_gpu{gpu}.log 2>&1"
    procs[gpu] = subprocess.Popen(["bash", "-c", cmd], cwd=CODE, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                                                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    print(f"GPU {gpu}: {conds}")
'''),
    ("code", r'''
# S4. Monitor until both workers finish.
def tail(path, n=2):
    try:
        return "".join([l for l in open(path) if l.strip() and "it/s]" not in l and "%|" not in l][-n:]).strip()
    except FileNotFoundError:
        return "(not started)"
t0 = time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(300)
    print(f"--- {(time.time() - t0) / 60:.0f} min")
    for g in procs: print(f"  GPU{g}: {tail(WORK / f't1_eval_gpu{g}.log')}")
for g, p in procs.items():
    CHK.check(p.returncode == 0, f"evaluation worker GPU {g} exited cleanly", 0, p.returncode)
    if p.returncode: print(tail(WORK / f"t1_eval_gpu{g}.log", 40))
'''),
    ("code", r'''
# S5. Intended vs actual for each condition.
from task1.t1_eval import EXPECTED_N
ALL_DONE = all((EVAL / c / "DONE").exists() for c in ORDER)
for c in ORDER:
    d = EVAL / c
    if not (d / "metrics.json").exists():
        done = [st for st in ["gsm8k_part.json", "mc_part.json", "qual_responses.jsonl"] if (d / st).exists()]
        CHK.check(False, f"{c}: metrics present", "metrics.json", f"missing (stages saved so far: {done or 'none'})"); continue
    m = read_json(d / "metrics.json")
    print(f"\n======== {c} ========")
    CHK.check(m["n_examples"] == EXPECTED_N, f"{c}: complete splits evaluated", "1319 / 10042 / 3270 / 2376", m["n_examples"])
    CHK.check(m["gsm8k_tracked_generations"] == 1319, f"{c}: every GSM8K generation has an exact stop reason", 1319, m["gsm8k_tracked_generations"])
    if c == "baseline":
        print("first GSM8K prompt exactly as the model saw it:\n" + repr(m["first_gsm8k_prompt"]))
        CHK.check(m["first_gsm8k_prompt"].endswith("<think>\n\n</think>\n\n"), "GSM8K prompt uses the non-thinking chat template (empty think block)",
                  "ends with '<think>\\n\\n</think>\\n\\n'", repr(m["first_gsm8k_prompt"][-25:]))
        CHK.check(m["lm_eval_config"]["gsm8k"]["num_fewshot"] == 0, "GSM8K is zero-shot", 0, m["lm_eval_config"]["gsm8k"]["num_fewshot"])
        CHK.observe("GSM8K generation settings used", m["lm_eval_config"]["gsm8k"]["generation_kwargs"])
    if "merge_check" in m:
        mc = m["merge_check"]
        CHK.check(mc["argmax_agreement"] > 0.98, f"{c}: merged vs unmerged logits agree", "argmax agreement > 0.98, small |diff| (FP16 rounding)",
                  f"agreement {mc['argmax_agreement']:.4f}, max |diff| {mc['max_abs_logit_diff']:.3f}, mean {mc['mean_abs_logit_diff']:.4f}")
    CHK.observe(f"{c}: primary scores (%)", {k: round(100 * m[k], 2) for k in ["gsm8k_flexible_em", "hellaswag_acc_norm", "boolq_acc", "arc_easy_acc_norm"]})
    CHK.observe(f"{c}: batch sizes used (GSM8K / likelihood tasks)", (m.get("gsm8k_batch_size"), m.get("mc_batch_size")))
    CHK.observe(f"{c}: GSM8K stop reasons / unextractable / mean tokens",
                (m["gsm8k_stop_reasons"], f"{100 * m['gsm8k_unextractable_rate']:.1f}%", round(m["gsm8k_mean_gen_tokens"], 1)))
    q = read_jsonl(d / "qual_responses.jsonl") if (d / "qual_responses.jsonl").exists() else []
    CHK.check(len(q) == 24, f"{c}: 24 qualitative responses saved", 24, len(q))
'''),
    ("code", r'''
# S6. The probe prompts must match lm-eval's prompts exactly; then the diagnostics (down_proj, layers 0/13/27, FP32).
if not ALL_DONE:
    print("SKIPPED: not every condition finished (see S4/S5). Run this notebook again with THIS version's output attached:"
          " finished conditions and stages are reused.")
else:
  from transformers import AutoTokenizer
  from task1.t1_diag import check_probe_formats
  tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
  check_probe_formats(tok, read_json(PROBE), EVAL / "baseline", CHK)
  r = subprocess.run([sys.executable, "-m", "task1.t1_diag", "--probe", PROBE, "--out", str(WORK / "t1_diag")], cwd=CODE,
                   capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
  print(r.stdout[-12000:]); print(r.stderr[-3000:] if r.returncode else "")
  CHK.check(r.returncode == 0 and "0 FAIL" in r.stdout, "diagnostics finished and their own checks passed", "0 FAIL", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.returncode)
'''),
    ("code", r'''
# S7. Tables, paired tests, figures and the 42-response annotation sheet.
if not ALL_DONE:
    print("SKIPPED: evaluation incomplete (see S5).")
else:
    from task1 import t1_report as R
    import pandas as pd
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    REP = WORK / "t1_report"; REP.mkdir(exist_ok=True)
    bench, comp, rows = R.benchmark_table(REP)
    print(bench.round(2).to_string(index=False)); print(); print(comp.round(2).to_string(index=False)); print()
    print(R.paired_tests(REP).round(4).to_string(index=False)); print()
    R.training_figures(REP)
    print(R.resource_table(REP).to_string(index=False))
    R.diag_figure(REP, WORK / "t1_diag")
    sub, summ, (first, last) = R.annotation_sheet(REP)
    CHK.check(len(sub) == 42, "annotation subset = first + last question x 3 formats x 7 conditions", 42, len(sub))
    print(f"\nannotate '{first}' and '{last}'  ->  t1_report/t1_annotation_42.xlsx"); print(summ.round(3).to_string())
    for a, b in [("baseline", "full_final"), ("baseline", "lora16_final"), ("lora16_final", "oplora_final"), ("full_mid", "full_final")]:
        CHK.observe(f"GSM8K per-question transitions {a} -> {b}", R.gsm8k_transitions(a, b))
'''),
    ("code", r'''
# S8. Look at the figures, then one aligned comparison (change QUESTION / FORMAT freely).
if not ALL_DONE:
    print("SKIPPED: evaluation incomplete (see S5).")
else:
    from IPython.display import Image, display
    for f in ["t1_loss_lr.png", "t1_l2sp.png", "t1_diagnostics.png"]:
        if (REP / f).exists(): display(Image(str(REP / f)))
    QUESTION, FORMAT = first, "json"
    R.aligned(QUESTION, FORMAT)
'''),
    ("code", r'''
# S9. Machine-readable results for the GitHub repo (no model weights): t1_results.zip in the output.
import zipfile
keep_ext = {".json", ".jsonl", ".csv", ".png", ".xlsx", ".log", ".yaml"}
roots = [WORK / "t1_eval", WORK / "t1_diag", WORK / "t1_report"] + [find_run(m) for m in ["full", "l2sp", "lora4", "lora16", "oplora"]]
data_dir = Path(PROBE).parent
with zipfile.ZipFile(WORK / "t1_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in [data_dir / "t1_train_ids.json", data_dir / "t1_probe_set.json"]:
        z.write(f, f"t1_data/{f.name}")
    for root in [r for r in roots if r is not None and Path(r).exists()]:
        for f in Path(root).rglob("*"):
            if f.is_file() and f.suffix in keep_ext and "recovery_u" not in str(f) and "export_" not in str(f):
                arc = f"t1_runs/{root.name}/{f.relative_to(root)}" if root.parent.name == "t1_runs" else f"{root.name}/{f.relative_to(root)}"
                z.write(f, arc)
CHK.observe("t1_results.zip", human_bytes((WORK / "t1_results.zip").stat().st_size), note="download it and commit its contents under results/task1/")
CHK.summary(); CHK.save(WORK / "t1_eval_checks.json")
print("\nNEXT:", "Task 1 evaluation complete: send S5-S8 output and t1_results.zip." if ALL_DONE else
      "not finished. Run again with pa3-task1, the training output AND this version's output attached; finished stages are reused.")
'''),
]

t1_recheck = [
    ("markdown", r'''
# PA3 · Task 1 · Redo of the OPLoRA Proposition-2 check (FP64 reference)

**Why this notebook exists.** The diagnostics in `t1_eval` reported one FAIL: "top-128 singular triples preserved, (W₀ + ΔW) vᵢ = σᵢ uᵢ" in layers 0 and 13 (1.17 × 10⁻⁴ and 1.08 × 10⁻⁴ against a 10⁻⁴ threshold). That check compared the trained matrix with a yardstick, a fresh FP32 SVD of W₀ computed on the GPU, and the yardstick itself is only accurate to about 10⁻⁴. The error it reported is the yardstick's error plus the update's effect.

**What it does.** Redoes the check with an exact yardstick (an FP64 SVD, accurate to about 10⁻¹⁵), with the same 10⁻⁴ threshold, on all 196 matrices OPLoRA adapted (the three diagnostic matrices included). On the GPU it also replays the original check and splits it into its two parts. It trains and evaluates nothing and changes none of the reported diagnostics; its results go to their own folder, `t1_recheck/`.

**Settings:** Accelerator **GPU T4** (×1 is enough; the GPU only replays the original check), Internet **On** (downloads Qwen3-0.6B). About 10–20 min.
**Attach:** `pa3-task1` (code version 2026.10.31 or newer) and the **t1_train output in which OPLoRA finished** (the same one you attached to `t1_eval`). Optional: the **t1_eval output**, so the original check's values are shown next to the replay.

**Run with "Save Version → Save & Run All".**

**How to read the output:** every check prints `PASS / FAIL / OBSERVE`, then the **expected** and **actual** lines.
'''),
    ("code", SETUP.replace('REQUIRED_CODE_VERSION = "2026.10.30.2"', 'REQUIRED_CODE_VERSION = "2026.10.31"')),
    ("code", r'''
# S2. Find the finished OPLoRA run (the newest finished copy among the attached outputs) and confirm it is the reported run.
from task1.t1_eval import find_run
WORK = Path("/kaggle/working"); OUT = WORK / "t1_recheck"
RUN = find_run("oplora")
CHK.check(RUN is not None and (RUN / "adapter_final_projected.pt").exists(), "finished OPLoRA run attached, with its projected factors",
          "t1_runs/oplora/adapter_final_projected.pt", RUN)
if RUN is not None:
    inj, fs = read_json(RUN / "injection.json"), read_json(RUN / "final_summary.json")
    CHK.check("re-orthonormalisation" in (inj.get("basis_method") or ""), "this is the reported OPLoRA run (re-orthonormalised bases)",
              "re-orthonormalised", inj.get("basis_method"))
    CHK.observe("its training-time constraint check on all 196 matrices: ‖Uₖᵀ ΔW‖ / ‖ΔW‖ and ‖ΔW Vₖ‖ / ‖ΔW‖",
                (f"{fs['oplora_worst_rel_UkT_dW']:.2e}", f"{fs['oplora_worst_rel_dW_Vk']:.2e}"), note="the reported run: 1.01e-07 and 1.19e-07")
orig = sorted(inputs("/kaggle/input/**/t1_diag/diag_checks.json"))   # the original diagnostics checks, from the attached t1_eval output
ORIG = Path(orig[-1]) if orig else Path("/nonexistent")
CHK.observe("original diagnostics checks (for side-by-side comparison)", str(ORIG) if orig else "not attached (optional)",
            note="read from the attached t1_eval output")
'''),
    ("code", r'''
# S3. The recheck, in a separate process. The FP64 work runs on the CPU; the GPU only replays the original FP32 check.
t0 = time.time()
r = subprocess.run([sys.executable, "-u", "-m", "task1.t1_recheck", "--out", str(OUT), "--original_checks", str(ORIG)],
                   cwd=CODE, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
print(r.stdout[-15000:]); print(r.stderr[-3000:] if r.returncode else "")
RECHECK_OK = r.returncode == 0 and "0 FAIL" in r.stdout
CHK.check(RECHECK_OK, "recheck finished and all its checks passed", "0 FAIL",
          [l for l in r.stdout.splitlines() if l.startswith("====")][-1:] or r.returncode)
CHK.observe("recheck time", f"{(time.time() - t0) / 60:.1f} min")
'''),
    ("code", r'''
# S4. Results for the GitHub repo: t1_recheck.zip (a few kB). Unzip it into results/task1/ (it creates results/task1/t1_recheck/).
import zipfile
files = sorted(OUT.glob("*")) if OUT.exists() else []
with zipfile.ZipFile(WORK / "t1_recheck.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f"t1_recheck/{f.name}")
CHK.observe("t1_recheck.zip contents", [f.name for f in files], note="download it and unzip it into results/task1/")
CHK.summary(); CHK.save(WORK / "t1_recheck_notebook_checks.json")
print("\nNEXT:", "send this notebook (S2-S4 output) and t1_recheck.zip." if RECHECK_OK else
      "a check failed: send this notebook's output as it is, so we can see which matrix and by how much.")
'''),
]

if __name__ == "__main__":
    nb(t1_train, ROOT / "notebooks" / "t1_train.ipynb")
    nb(t1_eval, ROOT / "notebooks" / "t1_eval.ipynb")
    nb(t1_recheck, ROOT / "notebooks" / "t1_recheck.ipynb")
    print("ok")
