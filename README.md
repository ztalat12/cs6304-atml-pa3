# ATML PA3 (EE-5102 / CS-6304, Fall 2026)

Programming Assignment 3, organised task by task. Each task has its own folder, notebooks and README with reproduction
steps, settings, decisions and attribution.

| Task | Folder | Notebooks | Status |
|---|---|---|---|
| 1. LLM fine-tuning: adaptation vs retention | [`task1/`](task1/README.md) | `notebooks/t1_train.ipynb`, `notebooks/t1_eval.ipynb`, `notebooks/t1_recheck.ipynb` | complete; results in `results/task1/` |
| 2. Vision-language models and visual grounding | [`task2/`](task2/README.md) | `notebooks/t2_vlm.ipynb` | complete; results in `results/task2/` |
| 3. Test-time compute | [`task3/`](task3/README.md) | `notebooks/t3_offline.ipynb`, `notebooks/t3_live.ipynb` | complete; results in `results/task3/` |

```
common/          helpers shared by all tasks (checks, seeding, I/O, LoRA / OPLoRA, model loading)
task1/ … task3/  each task's code and README
notebooks/       Kaggle notebooks
tests/           offline CPU tests (tiny random Qwen3 / SigLIP, synthetic data)
results/taskN/   machine-readable results: the notebooks' original output files (unchanged) plus tables rebuilt from them
requirements.txt environment specification
```

Every notebook prints `PASS / FAIL / OBSERVE` lines with expected and actual values. Model weights, checkpoints and raw
datasets are not committed (see `.gitignore`); selected example IDs and all reported numbers are in `results/`.
