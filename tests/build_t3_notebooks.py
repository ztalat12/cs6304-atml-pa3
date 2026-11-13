"""Builds notebooks/t3_offline.ipynb (CPU) and notebooks/t3_live.ipynb (GPU)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_nb import nb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SETUP = r'''
# S1. Find the Task 3 code (attached Kaggle dataset, or a clone of your GitHub repo) and copy it to /tmp.
import os, sys, glob, json, time, shutil, subprocess
from pathlib import Path
SESSION_START = time.time()     # Kaggle's 12-hour limit counts from about now

REPO_URL = ""        # optional: your public GitHub repo URL, used only if no code dataset is attached
REQUIRED_CODE_VERSION = "2026.11.03.2"   # this notebook needs this code version or newer
NEED_GPU = __NEED_GPU__
CODE = Path("/tmp/pa3_task3_code")

def code_version(root):
    import re
    try:
        m = re.search(r'CODE_VERSION\s*=\s*"([^"]+)"', (Path(root) / "common" / "__init__.py").read_text())
        return m.group(1) if m else "0 (no version stamp)"
    except FileNotFoundError:
        return "0 (no version stamp)"

hits = sorted({Path(p).parents[1] for p in glob.glob("/kaggle/input/**/task3/t3_offline.py", recursive=True)},
              key=code_version, reverse=True)
for h in hits:
    print(f"found code copy: {h}  (version {code_version(h)})")
if CODE.exists():
    shutil.rmtree(CODE)
if hits:
    if code_version(hits[0]) < REQUIRED_CODE_VERSION:
        raise SystemExit(f"The newest attached code is version {code_version(hits[0])}, but this notebook needs "
                         f"{REQUIRED_CODE_VERSION}. Upload the latest pa3-task3.zip as a new dataset version and re-attach it.")
    shutil.copytree(hits[0], CODE)
    print("using:", hits[0])
elif REPO_URL:
    subprocess.run(["git", "clone", "-q", REPO_URL, str(CODE)], check=True)
    print("code cloned from", REPO_URL)
else:
    raise SystemExit("Attach the pa3-task3 dataset (or set REPO_URL) and re-run this cell.")
sys.path.insert(0, str(CODE))
os.chdir(CODE)
CODE_COPIES = [str(h) for h in hits]
def inputs(pattern):
    """Files matching `pattern` in the ATTACHED inputs only (never inside a copy of the code)."""
    return [p for p in glob.glob(pattern, recursive=True) if not any(p.startswith(c + "/") for c in CODE_COPIES)]
from common.utils import Checks, human_bytes, dir_size_bytes, read_json, read_jsonl
CHK = Checks()
CHK.check(code_version(CODE) >= REQUIRED_CODE_VERSION, "code version in use", f">= {REQUIRED_CODE_VERSION}", code_version(CODE))
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    GPUS = [l.strip() for l in smi.stdout.strip().splitlines() if l.strip()]
except FileNotFoundError:
    GPUS = []
if NEED_GPU:
    CHK.check(len(GPUS) >= 1, "GPU visible", "GPU T4 (Settings -> Accelerator)", GPUS or "no GPU: is the accelerator switched on?")
else:
    CHK.observe("accelerator", GPUS or "CPU only (enough for this notebook)")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "math-verify==0.9.0"], check=True)
import importlib.metadata as md
vers = {p: md.version(p) for p in ["torch", "transformers", "numpy", "pandas", "pyarrow", "scipy", "math-verify"]}
CHK.observe("library versions", vers)
CHK.check(vers["math-verify"] == "0.9.0", "answer-equivalence scorer pinned (math-verify)", "0.9.0", vers["math-verify"])
# the supplied rollouts, attached as a Kaggle dataset (ATML-PA3-Data, unchanged)
from task3 import t3_data as D
FOLDERS = D.find_folders(["/kaggle/input"])
CHK.check(set(FOLDERS) == {"math", "mathvista"}, "supplied data attached (qwen25-math/, qwen25vl-mathvista/)", "both folders",
          {k: str(v) for k, v in FOLDERS.items()} or "not found: attach your atml-pa3-data dataset")
DATA_ROOT = str(Path(FOLDERS["math"]).parent) if "math" in FOLDERS else "/kaggle/input"
for k, v in FOLDERS.items():
    CHK.observe(f"{k} files", {f.name: human_bytes(f.stat().st_size) for f in sorted(Path(v).glob("*.parquet"))})
WORK = Path("/kaggle/working"); T3 = WORK / "t3"; T3.mkdir(parents=True, exist_ok=True)
'''

t3_offline = [
    ("markdown", r'''
# PA3 · Task 3 · Offline experiments on the supplied rollouts (Steps 1, 2, 3, 5, 6)

**What this notebook does.**
1. Loads both supplied folders, checks that every file joins and every token list lines up, and compares the totals with the supplied README.
2. Splits each dataset (sorted IDs, `default_rng(6304)`): 200 development + 800 held-out questions; makes the ten random orders of the 32 rollouts.
3. Groups and grades every answer: `math-verify` 0.9.0 for MATH, MathVista's official rules (with the released testmini metadata) for MathVista; missing answers → `NO_ANSWER`.
4. Runs every selection rule at N = 1, 2, 4, 8, 16, 32 over the ten orders; oracle coverage; the sampling-vs-selection decomposition; paired bootstrap intervals.
5. Signal analysis at N = 32: length / score / correctness associations, confidence–vision correlation, tokenwise vs rollout-level fusion.
6. Adaptive stopping: thresholds chosen on development, frozen, then evaluated on held-out.

**Settings:** Accelerator **None (CPU)** is enough. Internet **On** (installs `math-verify`, downloads MathVista's question metadata). About 20–40 min.
**Attach:** `pa3-task3` and your **`atml-pa3-data`** dataset (the Drive folder, uploaded unchanged). To rerun only the analysis, also attach this notebook's earlier output: the graded rollouts in `t3/prep/` are reused (saves about 15 min).

**Run with "Save Version → Save & Run All".** **How to read the output:** `PASS / FAIL / OBSERVE`, then the **expected** and **actual** lines.
'''),
    ("code", SETUP.replace("__NEED_GPU__", "False")),
    ("code", r'''
# S2. Preparation: load, check, split, order, group and grade (or reuse an attached earlier preparation).
prev = sorted(inputs("/kaggle/input/**/t3/prep/scorer.json"))
if (T3 / "prep" / "scorer.json").exists():
    print("preparation already in the working directory")
elif prev:
    shutil.copytree(Path(prev[-1]).parent, T3 / "prep"); print("reused the attached preparation:", Path(prev[-1]).parent)
else:
    meta = sorted(inputs("/kaggle/input/**/t3/prep/mathvista_meta.json"))
    cmd = [sys.executable, "-u", "-m", "task3.t3_offline", "prep", "--data_root", DATA_ROOT, "--out", str(T3)] + \
          (["--mathvista_meta", meta[-1]] if meta else [])
    r = subprocess.run(cmd, cwd=CODE, capture_output=True, text=True)
    print(r.stdout[-20000:]); print(r.stderr[-4000:] if r.returncode else "")
    CHK.check(r.returncode == 0 and "0 FAIL" in r.stdout, "preparation finished and its checks passed", "0 FAIL",
              [l for l in r.stdout.splitlines() if l.startswith("====")][-1:] or r.returncode)
CHK.observe("scorer (fixed before any evaluation)", read_json(T3 / "prep" / "scorer.json"))
'''),
    ("code", r'''
# S3. Analysis: every rule x budget x order, signals, adaptive stopping (thresholds from development only).
#     One process per dataset, so a problem in one cannot hide the other's results.
ANALYSIS_OK = True
for name in ["math", "mathvista"]:
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", "-m", "task3.t3_offline", "analyze", "--out", str(T3), "--datasets", name],
                       cwd=CODE, capture_output=True, text=True)
    print(r.stdout[-30000:])
    if r.returncode:
        print(f"return code {r.returncode}" + (" (killed: out of memory?)" if r.returncode in (-9, 137) else "")); print(r.stderr[-4000:])
    ok = r.returncode == 0 and "0 FAIL" in r.stdout
    ANALYSIS_OK &= ok
    CHK.check(ok, f"{name}: analysis finished and its checks passed", "0 FAIL",
              [l for l in r.stdout.splitlines() if l.startswith("==== ") and "PASS" in l][-1:] or f"return code {r.returncode}")
    CHK.observe(f"{name}: analysis time", f"{(time.time() - t0) / 60:.1f} min")
'''),
    ("code", r'''
# S4. Tables and figures (built only from the saved results).
from task3 import t3_report as R
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
REP = T3 / "report"
R.build(T3, REP)
for n in [n for n in ["math", "mathvista"] if (REP / f"t3_{n}_decomposition.csv").exists()]:
    print(f"\n==== {n}: sampling vs selection (held-out, % of questions)")
    print(pd.read_csv(REP / f"t3_{n}_decomposition.csv").round(1).to_string(index=False))
    print(f"\n==== {n}: signals at N = 32 (within question)")
    print(pd.read_csv(REP / f"t3_{n}_signals.csv").round(3).to_string(index=False))
    print(f"\n==== {n}: adaptive stopping (held-out)")
    print(json.dumps(read_json(T3 / "analysis" / f"{n}_adaptive_chosen.json"), indent=1))
    print(pd.read_csv(REP / f"t3_{n}_adaptive_heldout.csv").round(3).to_string(index=False))
    print(json.dumps(read_json(T3 / "analysis" / f"{n}_early_stops.json"), indent=1)[:3000])
from IPython.display import Image, display
for f in ["t3_accuracy_curves.png", "t3_failure_decomposition.png", "t3_mathvista_signals.png"]:
    if (REP / f).exists(): display(Image(str(REP / f)))
'''),
    ("code", r'''
# S5. Illustrative cases (stated rule: the first held-out questions, in split order, showing each behaviour; order 0).
for n in [n for n in ["math", "mathvista"] if (T3 / "analysis" / f"{n}_examples.json").exists()]:
    ex = read_json(T3 / "analysis" / f"{n}_examples.json")
    print(f"\n==== {n}: selection disagreements (self-consistency vs the main quality rule)")
    print(json.dumps(ex["selection_disagreement"], indent=1, ensure_ascii=False)[:4000])
    print(f"\n==== {n}: premature stops (quality-aware controller)")
    print(json.dumps(ex["premature_stop"], indent=1, ensure_ascii=False)[:4000])
'''),
    ("code", r'''
# S6. Machine-readable results for the GitHub repo: t3_offline_results.zip (no copies of the supplied data).
import zipfile
skip = {"mathvista_tokens.npz"}                       # per-token arrays copied from the supplied files: reproducible, not needed
with zipfile.ZipFile(WORK / "t3_offline_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(T3.rglob("*")):
        if f.is_file() and f.name not in skip and "hf_cache" not in str(f):
            z.write(f, str(f.relative_to(WORK)))
CHK.observe("t3_offline_results.zip", human_bytes((WORK / "t3_offline_results.zip").stat().st_size))
CHK.summary(); CHK.save(WORK / "t3_offline_notebook_checks.json")
print("\nNEXT:", "send this notebook (S2-S5 output) and t3_offline_results.zip." if ANALYSIS_OK else
      "something failed: send this notebook's output as it is.")
'''),
]

t3_live = [
    ("markdown", r'''
# PA3 · Task 3 · Live experiment: one long response vs parallel sampling vs sequential refinement (Step 4)

**What this notebook does.** On the fixed 50 MATH questions (the first 50 held-out questions of the split), with the same 3,072-token output allowance per question:

| Strategy | Generation |
|---|---|
| Single longer response | one response, ≤ 3,072 new tokens |
| Parallel sampling | three independent responses, ≤ 1,024 each; self-consistency picks the answer |
| Sequential refinement | the first parallel response, then one critique-and-revision response (≤ 2,048) |

Model `Qwen/Qwen2.5-3B-Instruct` at the pinned revision from the supplied README, FP16, no quantization, batch size 1, temperature 0.7 / top-p 0.8 / top-k 20 / repetition penalty 1.0, independently seeded calls (seed = 6304 + 10 × question + call), all on **one GPU** so runtimes are comparable. Every call is saved as soon as it finishes.

**Settings:** Accelerator **GPU T4** (one GPU is used), Internet **On** (downloads the model). Roughly 1.5–4 h.
**Attach:** `pa3-task3` and your **`atml-pa3-data`** dataset. If a session stops early, run again with this notebook's earlier output attached as well: finished calls are reused.

**Run with "Save Version → Save & Run All".** **How to read the output:** `PASS / FAIL / OBSERVE`, then the **expected** and **actual** lines.
'''),
    ("code", SETUP.replace("__NEED_GPU__", "True")),
    ("code", r'''
# S2. Launch generation on GPU 0 (separate process); earlier calls from an attached output are reused.
STOP_AFTER_HOURS = 11.0
prev = sorted(inputs("/kaggle/input/**/t3/live/live_calls.jsonl"))
for p in prev: print("reusing earlier calls from", p, f"({sum(1 for _ in open(p))} calls)")
LOG = WORK / "t3_live.log"
cmd = [sys.executable, "-u", "-m", "task3.t3_live", "generate", "--data_root", DATA_ROOT, "--out", str(T3),
       "--stop_at", str(SESSION_START + STOP_AFTER_HOURS * 3600)] + (["--resume_from", *prev] if prev else [])
proc = subprocess.Popen(cmd, cwd=CODE, stdout=open(LOG, "w"), stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
print("started:", " ".join(cmd[3:5]))
'''),
    ("code", r'''
# S3. Monitor: prints progress every 10 minutes until generation ends.
def tail(n=2):
    try: return "".join([l for l in open(LOG) if l.strip() and "it/s]" not in l and "%|" not in l][-n:]).strip()
    except FileNotFoundError: return "(not started)"
t0, tp = time.time(), time.time()
while proc.poll() is None:
    time.sleep(30)
    if time.time() - tp >= 600:
        tp = time.time(); print(f"--- {(time.time() - t0) / 60:.0f} min: {tail()}")
print(tail(40))
CHK.check(proc.returncode == 0, "generation process exited cleanly", 0, proc.returncode)
PAUSED = (T3 / "live" / "PAUSED").exists()
if PAUSED: print("PAUSED at the time limit: run this notebook again with THIS version's output attached to finish.")
'''),
    ("code", r'''
# S4. Score the three strategies: accuracy, actual tokens, runtime, truncation / extraction, revision transitions.
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
r = subprocess.run([sys.executable, "-u", "-m", "task3.t3_live", "score", "--out", str(T3)], cwd=CODE, capture_output=True, text=True)
print(r.stdout[-6000:]); print(r.stderr[-3000:] if r.returncode else "")
LIVE_OK = r.returncode == 0 and "0 FAIL" in r.stdout
CHK.check(LIVE_OK, "all 50 questions scored", "0 FAIL", [l for l in r.stdout.splitlines() if l.startswith("====")][-1:] or r.returncode)
if LIVE_OK:
    print(pd.read_csv(T3 / "live" / "live_summary.csv").round(3).T.to_string())
    print(pd.read_csv(T3 / "live" / "live_bootstrap.csv").round(2).to_string(index=False))
    print(json.dumps(read_json(T3 / "live" / "live_transitions.json"), indent=1))
    CHK.observe("environment", read_json(T3 / "live" / "live_env.json"))
'''),
    ("code", r'''
# S5. Illustrative refinement cases (stated rule: the first questions, in split order, where the revision changed the answer).
if LIVE_OK:
    pq = pd.read_csv(T3 / "live" / "live_per_question.csv")
    ch = pq[pq.revision_changed_answer].head(4)
    calls = {(c["sample_id"], c["call"]): c for c in read_jsonl(T3 / "live" / "live_calls.jsonl")}
    for _, x in ch.iterrows():
        a, b = calls[(x.sample_id, "parallel_1")], calls[(x.sample_id, "revision")]
        print(f"\n######## {x.sample_id}  gold {x.gold}  initial {x.initial_answer} ({'right' if x.initial_correct else 'wrong'}) -> "
              f"revised {x.revised_answer} ({'right' if x.revised_correct else 'wrong'})")
        print("INITIAL (end):", a["response"][-500:].replace("\n", " "))
        print("REVISION (start):", b["response"][:700].replace("\n", " "))
'''),
    ("code", r'''
# S6. Machine-readable results for the GitHub repo: t3_live_results.zip.
import zipfile
with zipfile.ZipFile(WORK / "t3_live_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted((T3 / "live").rglob("*")):
        if f.is_file(): z.write(f, str(f.relative_to(WORK)))
CHK.observe("t3_live_results.zip", human_bytes((WORK / "t3_live_results.zip").stat().st_size))
CHK.summary(); CHK.save(WORK / "t3_live_notebook_checks.json")
print("\nNEXT:", "send this notebook (S3-S5 output) and t3_live_results.zip." if LIVE_OK else
      "not finished: run again with pa3-task3, atml-pa3-data AND this version's output attached.")
'''),
]

if __name__ == "__main__":
    nb(t3_offline, ROOT / "notebooks" / "t3_offline.ipynb")
    nb(t3_live, ROOT / "notebooks" / "t3_live.ipynb")
    print("ok")
