"""Builds notebooks/t2_vlm.ipynb."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_nb import nb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETUP = r'''
# S1. Find the Task 2 code (attached Kaggle dataset, or a clone of your GitHub repo), copy it to /tmp, check the GPUs.
# All GPU work runs in separate processes, so this notebook itself never holds GPU memory.
import os, sys, glob, json, time, shutil, subprocess
from pathlib import Path
SESSION_START = time.time()     # Kaggle's 12-hour limit counts from about now

REPO_URL = ""        # optional: your public GitHub repo URL, used only if no code dataset is attached
REQUIRED_CODE_VERSION = "2026.10.30"   # this notebook needs this code version or newer
CODE = Path("/tmp/pa3_task2_code")

def code_version(root):
    import re
    try:
        m = re.search(r'CODE_VERSION\s*=\s*"([^"]+)"', (Path(root) / "common" / "__init__.py").read_text())
        return m.group(1) if m else "0 (no version stamp: older than 2026.10.30)"
    except FileNotFoundError:
        return "0 (no version stamp: older than 2026.10.30)"

# every attached copy of the code, newest version first (a dataset can hold several copies after re-uploads)
hits = sorted({Path(p).parents[1] for p in glob.glob("/kaggle/input/**/task2/t2_train.py", recursive=True)},
              key=code_version, reverse=True)
for h in hits:
    print(f"found code copy: {h}  (version {code_version(h)})")
if CODE.exists():
    shutil.rmtree(CODE)
if hits:
    if code_version(hits[0]) < REQUIRED_CODE_VERSION:
        raise SystemExit(f"The newest attached code is version {code_version(hits[0])}, but this notebook needs "
                         f"{REQUIRED_CODE_VERSION}. Upload the latest pa3-task2.zip as a new dataset version and re-attach it.")
    shutil.copytree(hits[0], CODE)
    print("using:", hits[0])
elif REPO_URL:
    subprocess.run(["git", "clone", "-q", REPO_URL, str(CODE)], check=True)
    print("code cloned from", REPO_URL)
else:
    raise SystemExit("Attach the pa3-task2 dataset (or set REPO_URL) and re-run this cell.")
sys.path.insert(0, str(CODE))
os.chdir(CODE)
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
assert "task2/t2_train.py" in SETUP and "pa3-task2" in SETUP

cells = [
    ("markdown", r'''
# PA3 · Task 2 · Vision-language model and visual grounding

**What this notebook does.**
1. Selects every image / question / pair with seed 6304 (COCO 2014 captions + instances, VQAv2 complementary pairs).
2. Downloads only the needed COCO images and caches frozen SigLIP features (196 × 768, FP16).
3. Shows you generated question pairs to inspect, then runs a pre-flight check of the real model.
4. Trains the three-stage curriculum (196 visual tokens on GPU 0; the optional 49-token run on GPU 1 at the same time).
5. Evaluates both final-test suites under four image conditions and builds every Task 2 table and figure.

**Time:** about 4–5 h on Kaggle **GPU T4 ×2** (each run's log prints its speed; the 49-token run is faster).
**Checkpoints and the 12-hour limit:** resumable checkpoints every 250 updates (plus one whenever an hour passes without a save); anything still training after `STOP_AFTER_HOURS` (11 h) saves and pauses. Run again with this output attached to continue: the selection, features and finished runs are reused, paused runs resume.

**Settings:** Accelerator **GPU T4 ×2**, Internet **On**.
**Attach:** `pa3-task2`. Only when continuing: also this notebook's earlier output (*Add Input → Your Work*).
**Run with "Save Version → Save & Run All".**

**How to read the output:** `PASS / FAIL / OBSERVE`, then the **expected** and **actual** lines.
'''),
    ("code", SETUP),
    ("code", r'''
# S2. CONFIG. The decisions (D-T2-*) are listed here; the values are the recommended ones.
WORK = Path("/kaggle/working/t2"); WORK.mkdir(parents=True, exist_ok=True)
IMG_DIR = Path("/tmp/coco_images")      # images are NOT saved to the output (only their features are)
RUN_COMPRESSION = True                  # D-T2-7: optional 49-token run on the second GPU, at the same time
STOP_AFTER_HOURS = 11.0                 # runs still training then pause themselves (Kaggle stops sessions at 12 h)
RERUN = []                              # e.g. ["196"] to retrain a run even if a FINISHED copy is attached
STOP_AT = SESSION_START + STOP_AFTER_HOURS * 3600
from task2 import t2_data as D, t2_model as M, t2_train as T
CHK.observe("D-T2-1 prompt wording", {"caption": M.CAPTION_INSTRUCTION, "after each question": M.ANSWER_SUFFIX.strip()})
CHK.observe("D-T2-2 label filters", {"presence 'yes' image: an instance covering at least": f"{D.CFG['min_area_frac_presence']:.0%} of the image",
                                     "counting: every instance at least": f"{D.CFG['min_area_frac_count']:.0%}, no crowd labels",
                                     "counts": D.CFG["count_range"]})
CHK.observe("D-T2-3 training pools and sampling", (D.CFG["grounding_pool_images"], T.SHARED["grounding_sampling"]))
CHK.observe("D-T2-4 captions per image", D.CFG["caption_choice"])
CHK.observe("D-T2-6 presence/counting pairs (dev, test)", (D.CFG["grounding_dev_pairs"], D.CFG["grounding_test_pairs"]))
CHK.observe("optimizer per stage", T.SHARED["optimizer"])
'''),
    ("code", r'''
# S3. Annotations + selection of every image / question / pair ID (CPU, about 5-10 min; reused if attached).
from collections import Counter
SEL = WORK / "t2_selection.json"
prev = glob.glob("/kaggle/input/**/t2/t2_selection.json", recursive=True)
if SEL.exists():
    print("selection already present")
elif prev:
    shutil.copy(prev[0], SEL); print("selection copied from", prev[0])
else:
    paths = D.fetch_annotations("/tmp/t2_ann")
    for k, v in paths.items(): print(f"  {k:16s} {v}")
    _, c2 = D.build_selection(paths, SEL)
    CHK.rows += c2.rows
sel = json.load(open(SEL))
for name, n in [("caption_train", 18000), ("caption_dev", 96), ("vqa_train", 1536), ("vqa_dev", 32), ("vqa_test", 96),
                ("grounding_dev", 24), ("grounding_test", 64)]:
    CHK.check(len(sel[name]) == n, f"{name} size", n, len(sel[name]))
for s in ["vqa_train", "vqa_dev", "vqa_test"]:
    CHK.observe(f"{s} pairs by answer type", dict(Counter(p[0]["type"] for p in sel[s])))
for s in ["grounding_dev", "grounding_test"]:
    CHK.observe(f"{s} pairs by kind", dict(Counter(p[0]["type"] for p in sel[s])))
CHK.observe("grounding training pools (pairs)", {k: len(v) for k, v in sel["grounding_train_pool"].items()})
CHK.observe("counting training answers", dict(sorted(Counter(q["answer"] for p in sel["grounding_train_pool"]["counting"] for q in p).items())))
CHK.observe("images needing features", len(sel["all_images"]))
'''),
    ("code", r'''
# S4. Images: only the needed ones (attached COCO 2014/2017 datasets are used if present). Then SigLIP features in a
#     separate process (so the encoder's GPU memory is released before Qwen is trained). Reused if attached.
from task2.t2_features import find_coco_roots, fetch_images
FEAT = WORK / "feats"
prevf = glob.glob("/kaggle/input/**/t2/feats/features_fp16.npy", recursive=True)
if not (FEAT / "features_fp16.npy").exists() and prevf:
    shutil.copytree(Path(prevf[0]).parent, FEAT, dirs_exist_ok=True); print("features copied from", Path(prevf[0]).parent)
need_all = not (FEAT / "features_fp16.npy").exists()
eval_keys = sorted({q["image"] for s in ["grounding_dev", "grounding_test", "vqa_dev", "vqa_test"] for p in sel[s] for q in p})
keys = sel["all_images"] if need_all else eval_keys          # images are needed for the features, or only for figures
t0 = time.time()
img_paths, failed = fetch_images(keys, IMG_DIR, roots=find_coco_roots())
CHK.check(not failed, f"every needed image available ({len(keys)})", 0, f"{len(failed)} missing, e.g. {failed[:3]}")
CHK.observe("image download time", f"{time.time() - t0:.0f} s")
if need_all:
    json.dump({k: str(v) for k, v in img_paths.items()}, open("/tmp/t2_image_paths.json", "w"))
    r = subprocess.run([sys.executable, "-m", "task2.t2_features", "--sel", str(SEL), "--paths", "/tmp/t2_image_paths.json",
                        "--out", str(FEAT)], cwd=CODE, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    print(r.stdout[-4000:]); print(r.stderr[-2000:] if r.returncode else "")
    CHK.check(r.returncode == 0, "feature caching finished, its own checks passed (shape, LayerNorm applied exactly once)", 0, r.returncode)
fi = read_json(FEAT / "features_index.json")
CHK.check(fi["shape"] == [len(sel["all_images"]), 196, 768] and fi["dtype"] == "float16" and fi["keys"] == sel["all_images"],
          "feature cache: one 196 x 768 FP16 block per selected image, in selection order", [len(sel["all_images"]), 196, 768], fi["shape"])
CHK.observe("feature caching cost", read_json(FEAT / "feature_cache_stats.json"))
'''),
    ("code", r'''
# S5. Inspect annotation-derived labels YOURSELF, independent of any model (COCO misses objects and has fuzzy counts).
#     Count how many of these 12 pairs look wrong or ambiguous: that number belongs in your discussion.
import matplotlib.pyplot as plt
from PIL import Image
pairs = sel["grounding_dev"][:3] + sel["grounding_dev"][12:15] + sel["grounding_test"][:3] + sel["grounding_test"][32:35]
fig, ax = plt.subplots(len(pairs), 2, figsize=(6.5, 3 * len(pairs)))
for i, pr in enumerate(pairs):
    for j, q in enumerate(pr):
        ax[i, j].imshow(Image.open(img_paths[q["image"]]).convert("RGB")); ax[i, j].axis("off")
        ax[i, j].set_title(f"{q['question']}  ->  {q['answer']}", fontsize=8)
plt.tight_layout(); plt.show()
'''),
    ("code", r'''
# S6. Pre-flight on the real models (separate process): prompt layout, frozen sets, connector gradients, calibration.
r = subprocess.run([sys.executable, "-m", "task2.t2_preflight", str(SEL), str(FEAT)], cwd=CODE, capture_output=True, text=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
pf = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1]) if r.returncode == 0 else None
if pf is None: print(r.stderr[-4000:])
CHK.check(pf is not None, "pre-flight ran", "JSON result", None if pf else "failed: see error above")
if pf:
    print("chat template with markers:\n" + repr(pf["rendered_template"]))
    print("text part of a training example (visual positions removed):\n" + repr(pf["decoded_text_part"]))
    CHK.check(pf["n_visual_positions"] == 196 and pf["first_visual_index"] == pf["n_prefix_tokens"] and pf["prefix_text"] == "<|im_start|>user\n",
              "196 visual positions right after '<|im_start|>user\\n', before the question", "196 at index after prefix",
              (pf["n_visual_positions"], pf["first_visual_index"], repr(pf["prefix_text"])))
    CHK.observe("placeholder token at the visual positions (its embedding is replaced)", pf["placeholder_token"])
    CHK.check(pf["supervised_text"] == "yes<|im_end|>", "loss only on the answer + <|im_end|>", "yes<|im_end|>", pf["supervised_text"])
    CHK.check(pf["decoded_text_part"].endswith("<think>\n\n</think>\n\nyes<|im_end|>"), "non-thinking chat structure kept", "...<think>\\n\\n</think>\\n\\nyes<|im_end|>",
              repr(pf["decoded_text_part"][-45:]))
    CHK.check(pf["no_image_equals_text_part"], "no-image input = same text tokens, visual positions removed", True, pf["no_image_equals_text_part"])
    CHK.check(pf["qwen_all_frozen"] and pf["connector_trainable"], "all Qwen parameters frozen, connector trainable", True,
              (pf["qwen_all_frozen"], pf["connector_trainable"]))
    CHK.check(pf["connector_params"] == 1_837_056, "connector parameters (768x1024 + 1024 + 1024x1024 + 1024)", "1,837,056", f"{pf['connector_params']:,}")
    CHK.check(all(g > 0 for g in pf["connector_grad_norms"]), "gradients reach the connector through frozen, checkpointed Qwen", "> 0",
              [round(g, 5) for g in pf["connector_grad_norms"]])
    CHK.check(pf["qwen_grads_none"], "no gradients on frozen Qwen weights", True, pf["qwen_grads_none"])
    CHK.check(pf["lora_params"] == 4_587_520 and pf["lora_modules"] == 112, "LoRA r16 on q/k/v/o in 28 layers", "4,587,520 in 112 matrices",
              f"{pf['lora_params']:,} in {pf['lora_modules']}")
    CHK.observe("D-T2-5 connector calibration (training data only)", pf["calibration"])
    CHK.observe("initial caption NLL (before any training)", round(pf["initial_caption_nll"], 3))
'''),
    ("code", r'''
# S7. Launch training: 196-token main run on GPU 0, optional 49-token run on GPU 1 (same seed, data, budgets).
RUNS = {196: WORK / "run196", 49: WORK / "run49"}
jobs = [196] + ([49] if RUN_COMPRESSION and len(GPUS) > 1 else [])
procs = {}
for gpu, ntok in enumerate(jobs):
    out = RUNS[ntok]
    if (out / "DONE").exists():
        print(f"{ntok}-token run already DONE"); continue
    prev_done = sorted(glob.glob(f"/kaggle/input/**/t2/run{ntok}/DONE", recursive=True))
    if prev_done and str(ntok) not in [str(x) for x in RERUN]:
        shutil.copytree(Path(prev_done[-1]).parent, out, dirs_exist_ok=True)
        print(f"{ntok}-token run finished in an earlier session, copied from {Path(prev_done[-1]).parent}"); continue
    prevc = sorted([p for p in glob.glob(f"/kaggle/input/**/t2/run{ntok}/ckpt_u*/state.pt", recursive=True)
                    if not (Path(p).parents[1] / "DONE").exists()], key=lambda p: int(Path(p).parent.name[6:]))
    extra = ["--resume_from", str(Path(prevc[-1]).parent)] if prevc and not out.exists() else []
    cmd = [sys.executable, "-u", "-m", "task2.t2_train", "--sel", str(SEL), "--feats", str(FEAT), "--out", str(out),
           "--tokens", str(ntok), "--stop_at", f"{STOP_AT:.0f}"] + extra
    procs[ntok] = subprocess.Popen(" ".join(cmd) + f" >> {WORK}/train{ntok}.log 2>&1", shell=True, cwd=CODE,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false"})
    print(f"GPU {gpu}: {ntok}-token run started {extra}")
'''),
    ("code", r'''
# S8. Monitor: checks every 30 s, prints the newest line of each run every 10 minutes.
def tail(p, n=1):
    try: return "".join([l for l in open(p) if l.strip() and "it/s]" not in l][-n:]).strip()
    except FileNotFoundError: return "(not started)"
t0, tp = time.time(), time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(30)
    if time.time() - tp >= 600:
        tp = time.time(); print(f"--- {(time.time() - t0) / 3600:.2f} h (session {(time.time() - SESSION_START) / 3600:.2f} h)")
        for n in procs: print(f"  [{n}] {tail(WORK / f'train{n}.log')}")
STATUS = {}
for n in jobs:
    o = RUNS[n]
    STATUS[n] = "DONE" if (o / "DONE").exists() else (f"PAUSED at update {read_json(o / 'PAUSED')['global_update']} / 4500"
                                                      if (o / "PAUSED").exists() else "STOPPED (crashed)")
    if STATUS[n] == "DONE": CHK.check(True, f"{n}-token training finished")
    elif STATUS[n].startswith("PAUSED"): CHK.observe(f"{n}-token training paused at the time limit", STATUS[n], note="run again with this output attached")
    else:
        CHK.check(False, f"{n}-token training finished", "DONE", "stopped: last log lines below"); print(tail(WORK / f"train{n}.log", 40))
'''),
    ("code", r'''
# S9. Intended vs actual for training.
for n in [j for j in jobs if STATUS.get(j) == "DONE"]:
    o = RUNS[n]
    tl, dl, st = read_jsonl(o / "train_log.jsonl"), read_jsonl(o / "dev_log.jsonl"), read_json(o / "stage_stats.json")
    print(f"\n======== {n} visual tokens ========")
    CHK.check(len(tl) == 4500 and tl[-1]["global_update"] == 4500, f"{n}: 4,500 updates (72,000 example presentations)", 4500, len(tl))
    CHK.check([d["global_update"] for d in dl] == T.dev_points(), f"{n}: development evaluations at 0, every 500 and stage ends",
              T.dev_points(), [d["global_update"] for d in dl])
    CHK.check([s["name"] for k, s in sorted(st.items())] == ["caption_alignment", "visual_instruction", "natural_vqa"], f"{n}: three stages")
    CHK.check(st["1"]["lora_params"] == 0 and st["2"]["lora_params"] == 4_587_520 == st["3"]["lora_params"],
              f"{n}: LoRA only in stages 2-3 (4,587,520)", "0 / 4,587,520 / 4,587,520", (st["1"]["lora_params"], st["2"]["lora_params"], st["3"]["lora_params"]))
    CHK.check(all((o / f"weights_after_stage{k}.pt").exists() for k in [1, 2, 3]), f"{n}: connector/adapter weights saved after each stage")
    for k, target in [("2", {"presence": .50, "counting": .25, "caption": .25}), ("3", {"vqa": .70, "presence": .10, "counting": .10, "caption": .10})]:
        d = st[k]["mix"]["draws"]; tot = sum(d.values())
        CHK.observe(f"{n}: stage {k} draw shares (target {target})", {kk: round(v / tot, 3) for kk, v in sorted(d.items())}, note=f"{tot} draws")
    CHK.observe(f"{n}: seconds / peak GPU GB / trainable params per stage",
                {k: (round(v["seconds"]), round(v["peak_allocated_gb"], 2), f"{v['trainable_params']:,}") for k, v in sorted(st.items())})
    CHK.observe(f"{n}: skipped updates (FP16 overflow)", sum(t["skipped"] for t in tl))
    CHK.observe(f"{n}: dev swap penalty dNLL at 0 / 2,250 / 3,750 / 4,500 (annotation-derived ; natural)",
                [(d["global_update"], round(d["grounding_dev_swap_penalty"], 3), round(d["vqa_dev_swap_penalty"], 3))
                 for d in dl if d["global_update"] in (0, 2250, 3750, 4500)])
    CHK.observe(f"{n}: connector scale used", read_json(o / "calibration.json")["scale"])
'''),
    ("code", r'''
# S10. Final evaluation on both test suites (4 image conditions), one process per finished run.
EV = {n: WORK / f"eval{n}" for n in jobs}
procs = {}
for gpu, n in enumerate([j for j in jobs if STATUS.get(j) == "DONE"]):
    prev = sorted(glob.glob(f"/kaggle/input/**/t2/eval{n}/DONE", recursive=True))
    if not (EV[n] / "DONE").exists() and prev and str(n) not in [str(x) for x in RERUN]:
        shutil.copytree(Path(prev[-1]).parent, EV[n], dirs_exist_ok=True); print(f"eval{n} copied from", Path(prev[-1]).parent)
    if (EV[n] / "DONE").exists(): continue
    cmd = f"{sys.executable} -u -m task2.t2_eval --sel {SEL} --feats {FEAT} --run {RUNS[n]} --out {EV[n]} --tokens {n} > {WORK}/eval{n}.log 2>&1"
    procs[n] = subprocess.Popen(cmd, shell=True, cwd=CODE, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
for n, p in procs.items():
    p.wait(); CHK.check(p.returncode == 0, f"{n}-token final evaluation finished", 0, p.returncode)
    print(open(WORK / f"eval{n}.log").read()[-5000:])
'''),
    ("code", r'''
# S11. Tables, curves, aligned complementary-image examples, resources (and the 49-token comparison).
from task2 import t2_report as R
import pandas as pd
from IPython.display import Image as IImage, display
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
REP = WORK / "report"; REP.mkdir(exist_ok=True)
if (EV[196] / "final_metrics.json").exists():
    R.learning_curves(RUNS[196], REP)
    df, adf, m196 = R.final_tables(EV[196], REP)
    print(df.round(2).to_string(index=False)); print(); print(adf.round(3).to_string(index=False))
    picks = R.select_examples(EV[196])
    ex = R.example_table(picks); ex.to_csv(REP / "t2_examples.csv", index=False); print(ex.to_string(index=False))
    fig_paths, _ = fetch_images(sorted({r["image"] for e in picks for r in e["rows"]["correct"]}), IMG_DIR, roots=find_coco_roots())
    R.example_figure(picks, fig_paths, REP)
    res = R.resource_table(RUNS[196], read_json(FEAT / "feature_cache_stats.json"), EV[196])
    res.to_csv(REP / "t2_resources.csv", index=False); print(res.to_string(index=False))
    for f in ["t2_learning_curves.png", "t2_examples.png"]:
        display(IImage(str(REP / f)))
    if 49 in EV and (EV[49] / "final_metrics.json").exists():
        R.learning_curves(RUNS[49], REP, tag="_49"); R.final_tables(EV[49], REP, tag="_49")
        comp = R.compression_table(m196, read_json(EV[49] / "final_metrics.json")); comp.to_csv(REP / "t2_compression.csv", index=False)
        print("\n196 vs 49 visual tokens:"); print(comp.round(2).to_string(index=False))
        r49 = R.resource_table(RUNS[49], read_json(FEAT / "feature_cache_stats.json"), EV[49]); r49.to_csv(REP / "t2_resources_49.csv", index=False)
        print(r49.to_string(index=False))
'''),
    ("code", r'''
# S12. Machine-readable results for the GitHub repo (no features, checkpoints or weights): t2_results.zip in the output.
import zipfile
keep_ext = {".json", ".jsonl", ".csv", ".png", ".log"}
with zipfile.ZipFile(Path("/kaggle/working") / "t2_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in WORK.rglob("*"):
        rel = f.relative_to(WORK)
        if f.is_file() and f.suffix in keep_ext and not str(rel).startswith(("feats/features_fp16", )) and "ckpt_u" not in str(rel):
            z.write(f, f"t2/{rel}")
CHK.observe("t2_results.zip", human_bytes((Path("/kaggle/working") / "t2_results.zip").stat().st_size), note="commit its contents under results/task2/")
total = dir_size_bytes(Path("/kaggle/working"))
CHK.check(total < 19.5e9, "working directory fits Kaggle's 20 GB output limit", "< 19.5 GB", human_bytes(total))
CHK.summary(); CHK.save(WORK / f"t2_checks_{time.strftime('%Y%m%d_%H%M')}.json")
if all(v == "DONE" for v in STATUS.values()):
    print("\nNEXT: Task 2 is complete. Send S9-S11 output and t2_results.zip.")
else:
    print("\nNEXT: not finished:", STATUS, "\nRun this notebook again with THIS version's output attached; it continues where it stopped.")
'''),
]

if __name__ == "__main__":
    nb(cells, ROOT / "notebooks" / "t2_vlm.ipynb")
    print("ok")
