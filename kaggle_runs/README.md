# Kaggle runs (with outputs)

The notebooks exactly as they ran on Kaggle, outputs included, downloaded after each session and committed unchanged
(byte-identical to the downloads). They are the record behind the numbers in `results/`. The clean, output-free
versions to rerun are in [`../notebooks/`](../notebooks/).

Only the final run of each notebook is here. Earlier attempts that were replaced are listed at the bottom with the reason.

## Task 1 · LLM fine-tuning (run in this order)

| Notebook | Template | What it did | Checks (final summary line) |
|---|---|---|---|
| [`task1/pa3-task1-run1.ipynb`](task1/pa3-task1-run1.ipynb) | `t1_train.ipynb` | built the data and trained **Full FT, L2-SP, LoRA-4 and LoRA-16** (the reported runs). Its OPLoRA run was not used: its constraint check failed, and OPLoRA was retrained in the next notebook | 45 PASS · 1 FAIL (that OPLoRA run) · 29 OBSERVE |
| [`task1/task1-oplora-run2.ipynb`](task1/task1-oplora-run2.ipynb) | `t1_train.ipynb` | retrained **OPLoRA (k = 128)** from scratch with the corrected code; the four finished runs were copied in from `pa3-task1-run1`, not retrained. This is the reported OPLoRA | 40 PASS · 0 FAIL · 24 OBSERVE |
| [`task1/task1-evaluation-r2.ipynb`](task1/task1-evaluation-r2.ipynb) | `t1_eval.ipynb` | evaluated all 7 conditions (benchmarks, McNemar + Holm, ρₖ / η / R(z) diagnostics, 24 qualitative prompts, the 42-response annotation sheet) | 39 PASS · 1 FAIL · 28 OBSERVE; the FAIL is the Proposition 2 check on layers 0 and 13, which compared against an FP32 GPU SVD |
| [`task1/pa3-t1-reeval.ipynb`](task1/pa3-t1-reeval.ipynb) | `t1_recheck.ipynb` | redid that Proposition 2 check in FP64: the OPLoRA update changes the top-128 singular triples by at most about 3 × 10⁻⁵, so the earlier FAIL came from the FP32 yardstick, not from OPLoRA | 5 PASS · 0 FAIL · 5 OBSERVE (plus two earlier 6 PASS summaries in the same run) |

## Task 2 · Vision-language model

| Notebook | Template | What it did | Checks |
|---|---|---|---|
| [`task2/pa3-task2-r1.ipynb`](task2/pa3-task2-r1.ipynb) | `t2_vlm.ipynb` | SigLIP + connector + Qwen3 with LoRA, three-stage training, the four image conditions, 196 vs 49 tokens, bootstrap intervals and figures (10.7 MB because the figures are embedded) | 54 PASS · 0 FAIL · 38 OBSERVE |

## Task 3 · Test-time compute

| Notebook | Template | What it did | Checks |
|---|---|---|---|
| [`task3/pa3-task3-offline.ipynb`](task3/pa3-task3-offline.ipynb) | `t3_offline.ipynb` | the offline experiments on the supplied rollouts (MATH and MathVista): budgets 1–32, every selection rule, coverage, signal analysis, adaptive stopping | no FAIL in any section; last summary 6 PASS · 0 FAIL · 8 OBSERVE |
| [`task3/pa3-task3-live.ipynb`](task3/pa3-task3-live.ipynb) | `t3_live.ipynb` | the live run on 50 MATH questions: single 3,072 / parallel 3 × 1,024 / refinement 1,024 + 2,048 | 6 PASS · 0 FAIL · 5 OBSERVE |

## Earlier attempts left out

| Kaggle notebook | Why it was replaced | Replaced by |
|---|---|---|
| `task1-oplora` | OPLoRA retraining that still ran the old code (37 PASS · 2 FAIL) | `task1-oplora-run2` |
| `task1-evaluation` | ran out of memory before finishing; no results | `task1-evaluation-r2` |
| `pa3-task3-offline` (first run) | MATH finished, but the MathVista analysis stopped before writing its results, so the report cell crashed (`FileNotFoundError`); fixed in code version 2026.11.03.2 | `pa3-task3-offline` (rerun) |
