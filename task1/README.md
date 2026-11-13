# PA3 · Task 1 · LLM fine-tuning: adaptation vs retention

Fine-tunes Qwen/Qwen3-0.6B on 40,000 MetaMathQA examples with five methods (Full FT, Full FT + L2-SP, LoRA-4, LoRA-16,
OPLoRA-16), evaluates seven conditions on GSM8K / HellaSwag / BoolQ / ARC-Easy, runs weight-space and activation
diagnostics, and generates the qualitative responses and annotation sheet.

## Files

| File | What it does |
|---|---|
| `t1_data.py` | selects the first 40,000 eligible MetaMathQA examples (source order), tokenises them, draws the activation-probe set |
| `t1_train.py` | trains one method per process; resumable; pauses itself at a time limit (`--stop_at`) |
| `t1_eval.py` | the 7 evaluation conditions: lm-eval benchmarks + 24 qualitative prompts each |
| `t1_diag.py` | ρₖ, η (weight space) and R(z) (activations) on `down_proj`, layers 0 / 13 / 27, in FP32 |
| `t1_recheck.py` | redo of the diagnostics' Proposition-2 check for OPLoRA against an exact FP64 reference, on all 196 adapted matrices (see Diagnostics self-check) |
| `t1_report.py` | benchmark table, paired McNemar tests, figures, resource table, 42-response annotation sheet + rubric |
| `first_plot_questions.yaml` | the 24 qualitative prompts (see Attribution) |
| `../notebooks/t1_train.ipynb` | Kaggle notebook: data preparation + all five training runs in one session |
| `../notebooks/t1_eval.ipynb` | Kaggle notebook: evaluation, diagnostics, tables, figures, `t1_results.zip` |
| `../notebooks/t1_recheck.ipynb` | Kaggle notebook: the FP64 Proposition-2 recheck, `t1_recheck.zip` (about 15 min) |
| `../common/` | shared helpers: PASS / FAIL / OBSERVE checks, seeding, I/O, LoRA + OPLoRA adapters, dtype-safe model loading |
| `../tests/test_task1.py` | 101 offline CPU checks on a tiny random Qwen3 |

## Reproducing (Kaggle, GPU T4 ×2, Internet on)

1. Upload this package as a Kaggle dataset named `pa3-task1` (or set `REPO_URL` in cell S1 of each notebook).
2. Run `notebooks/t1_train.ipynb` with **Save Version → Save & Run All**, attaching `pa3-task1`.
   GPU 0 trains Full FT then L2-SP; GPU 1 trains LoRA-4, LoRA-16, then OPLoRA. Estimated 6–9 h.
   Checkpoints and the 12-hour limit: besides the prescribed checkpoints (updates 312 / 625 / 937 / 1,250), a run saves an
   extra recovery checkpoint whenever an hour passes without a save. Anything still training after 11 h saves a checkpoint
   and pauses (exit code 3), so the notebook finishes before Kaggle's 12-hour limit and the output is kept. Running the
   notebook again with that output attached copies finished runs over and resumes the others from their newest checkpoint;
   resuming is bit-identical to an uninterrupted run (tested). Runs on the same GPU are independent: a crash in one does
   not stop the next.
3. Run `notebooks/t1_eval.ipynb`, attaching `pa3-task1` and the output of the last training session.
4. Download `t1_results.zip` from the evaluation output and unzip it into `results/task1/`.
5. Run `notebooks/t1_recheck.ipynb`, attaching `pa3-task1` and the training output in which OPLoRA finished (optionally
   also the evaluation output, to show the original check's values next to the replay); unzip `t1_recheck.zip` into
   `results/task1/`.

The notebooks read earlier runs and results only from attached notebook outputs, never from copies inside the code
dataset (cell S1 defines `inputs()`, which skips every attached copy of the code).

## Results layout (`results/task1/`)

| Folder | Contents |
|---|---|
| `t1_data/`, `t1_runs/`, `t1_eval/`, `t1_diag/`, `t1_report/` | the original files from the evaluation output (`t1_results.zip`), unchanged |
| `t1_recheck/` | the original files from the recheck output (`t1_recheck.zip`), unchanged |
| `t1_report_rebuilt/` | tables and figures regenerated from the original files above (no GPU), with the command below |

`python -m task1.t1_report --results_dir results/task1 --out results/task1/t1_report_rebuilt`

The rebuilt benchmark, resource and diagnostics numbers are identical to the originals. The rebuild adds analyses written
after the evaluation ran: BoolQ answer bias (`t1_boolq_bias.csv`), GSM8K answer format (`t1_gsm8k_format.csv`), OPLoRA
preservation including the FP64 recheck (`t1_oplora_preservation.csv`), a Holm-correction column in
`t1_paired_tests.csv`, distinguishable line styles in the figures, and a rubric tab in the annotation sheet. An existing,
possibly annotated `t1_annotation_42.xlsx` is never overwritten.

Offline tests (CPU, no downloads): `python tests/make_tiny.py /tmp/tinyqwen && python tests/test_task1.py /tmp/tinyqwen`

## Settings (as prescribed by the manual)

Qwen/Qwen3-0.6B with `enable_thinking=False`, seed 6304, every run starting from the original checkpoint.
One epoch of 40,000 examples = 1,250 updates of 32 (microbatch 2 × accumulation 16). AdamW (β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸),
weight decay 0.01 on matrices only, 38 warm-up updates then cosine decay, global-norm clipping 1.0, sequences ≤ 2,048 tokens,
gradient checkpointing, FP16 autocast with dynamic loss scaling, FP32 optimizer states (FP32 master weights for Full FT;
frozen FP16 backbone + FP32 adapters for LoRA / OPLoRA). Loss on response tokens and `<|im_end|>`, normalised by the number of
supervised tokens in the effective batch. Peak LR 2 × 10⁻⁵ (Full FT variants) and 2 × 10⁻⁴ (adapters); L2-SP λ = 10⁻³ over all
unique parameters; LoRA (r, α) = (4, 8) and (16, 32), dropout 0.05, random A, zero B, all seven projections in every layer;
OPLoRA r = 16, α = 32, k = 128, same initial A as LoRA-16 (verified by fingerprint). Recovery checkpoints at updates 312, 625,
937, 1,250. Evaluation with lm-evaluation-harness 0.4.13, zero-shot, complete splits, FP16, dropout off; batch size 32 for GSM8K generation and 8 for the likelihood tasks, each falling back to a smaller size on a GPU out-of-memory error (the size used is recorded in `metrics.json`): `gsm8k_cot` (non-thinking chat
template, greedy, 1,024 new tokens, flexible-extract exact match), HellaSwag (acc_norm), BoolQ (acc), ARC-Easy (acc_norm);
raw accuracy also reported.

## Decisions where the manual leaves a choice

| # | Decision | Choice |
|---|---|---|
| D1 | GSM8K-overlap "text normalisation" | Unicode NFKC → lowercase → non-alphanumeric characters to spaces → collapse spaces; exact match of `query` or `original_question` against any GSM8K test question |
| D2 | Cosine schedule convention | `transformers.get_cosine_schedule_with_warmup`: update 1 uses LR 0, linear to the peak at update 39, cosine to 0 at update 1,250 |
| D3 | Where L2-SP keeps θ₀ | pinned CPU memory in BF16 on GPUs under 20 GB (bit-exact, since the checkpoint is stored in BF16), copied to the GPU in chunks each update; storage size, transfer time and penalty time are logged |
| D4 | Annotation scales | format compliance yes / partial / no / n/a; coherence and relevance 1–3; repetition, unnecessary maths framing and inappropriate content yes / no; truncation automatic. Definitions are in the `rubric` tab of `t1_annotation_42.xlsx` |
| D5 | Session plan | one Kaggle session for all five runs (two GPUs in parallel), with a self-pause at 11 h as a safety net |

## Implementation notes

- Prompt and response are tokenised separately and concatenated; data preparation reports how often this differs from
  tokenising the joined string.
- GSM8K truncation flags are exact: a subclass of lm-eval's `HFLM` records, for every generation, whether it ended with the
  end-of-turn token, a stop string, or the 1,024-token cap.
- Disk (Kaggle keeps 20 GB of output): a full-model recovery checkpoint is ~7 GB, so only the newest is kept while a run
  trains, and a finished run deletes its recovery checkpoints. All four are written at the prescribed updates, and their sizes
  are logged in `checkpoints.jsonl`. Checkpoints are written atomically, so a crash mid-save never corrupts the newest one.
- Only Full FT is exported at the midpoint, since it is the only method evaluated there.
- Evaluation saves each stage of a condition (GSM8K, likelihood tasks, qualitative) as soon as it finishes, so an
  interrupted run resumes without redoing finished stages. A first attempt with batch size 32 for the likelihood tasks ran
  out of memory: lm-eval scores the longest requests first and keeps log-probabilities over the full 151,936-token
  vocabulary for every position in the batch.
- OPLoRA bases: after the exact FP32 SVD, Uₖ and Vₖ are re-orthonormalised once with a QR decomposition in FP64 (column
  signs kept, so each vector stays equal to its singular vector to ~10⁻⁷) and stored in FP32. A first run with the raw FP32
  SVD vectors held the constraints only to 1.9 × 10⁻⁴ relative error: their perpendicularity was off by ~1.3 × 10⁻⁵, and
  raw B drifts into the protected directions under AdamW, which magnifies that error (the drift is removed by the
  projection, so it never reaches the model). Re-orthonormalising brings the perpendicularity error to ~1.6 × 10⁻⁶.
  The reported OPLoRA run uses the re-orthonormalised bases and holds the constraints to 1.0 × 10⁻⁷ (output side) and
  1.2 × 10⁻⁷ (input side) across all 196 matrices, i.e. FP32 rounding level. Its training loss matches the first run
  (last-25-update mean 0.2034 vs 0.2033). `t1_train.ipynb` retrains the methods listed in `RERUN` and reuses the other
  finished runs from an attached output.
- Beyond the required evidence, `t1_report.py` reports McNemar's exact test on per-question flips for each controlled
  comparison (with a Holm correction over all 20 tests), BoolQ's share of "yes" predictions and per-class accuracy
  (`t1_boolq_bias.csv`), and how GSM8K answers are written (`t1_gsm8k_format.csv`; strict-match is 0% for every model
  because no model writes "The answer is X." without a colon).
- Diagnostics self-check: `diag_checks.json` reports two FAILs for "(W₀ + ΔW) vᵢ = σᵢ uᵢ" (relative error 1.17 × 10⁻⁴
  and 1.08 × 10⁻⁴ against a 10⁻⁴ threshold in layers 0 and 13). That check mixes in the diagnostic SVD's own accuracy:
  ‖W₀Vₖ − UₖΣₖ‖ does not involve the update at all. The update's own contribution is ‖ΔW Vₖ‖ = √(ρ input side) × ‖ΔW‖,
  at most 8.4 × 10⁻⁵ in absolute terms, while ‖W₀Vₖ‖ ≥ √(128/1024) × ‖W₀‖ ≈ 15–17, so the update changes what
  `down_proj` does on the protected input directions by at most 5 × 10⁻⁶ relative (`t1_oplora_preservation.csv`).
  The constraint check with OPLoRA's own training bases gives 1.0–1.2 × 10⁻⁷ (`final_summary.json`).
  `t1_recheck.py` redoes the check without the reference's error: the reference is an FP64 SVD of the BF16-exact W₀
  (W₀Vₖ = UₖΣₖ to ~10⁻¹⁵), ΔW = (α/r) B̃Ã is multiplied in FP64, and both sides of Proposition 2,
  (W₀ + ΔW)vᵢ = σᵢuᵢ and uᵢᵀ(W₀ + ΔW) = σᵢvᵢᵀ, are tested on all 196 adapted matrices with the same 10⁻⁴ threshold.
  On a GPU it also replays the original FP32 check and reports its reference part and update part separately.
  Results: `results/task1/t1_recheck/` (`oplora_prop2_recheck.csv`, `original_check_replay.csv`, `recheck_checks.json`).
  Outcome: 6 PASS, 0 FAIL. The FP64 reference satisfies W₀Vₖ = UₖΣₖ to 2.5 × 10⁻¹⁵. On the three diagnostic matrices,
  Proposition 2 holds to 1.4–3.8 × 10⁻⁶ (right side) and 2.5–5.0 × 10⁻⁶ (left side); over all 196 matrices the median is
  3.2 × 10⁻⁶ / 2.8 × 10⁻⁶ and the worst 3.0 × 10⁻⁵ (layer 17 `gate_proj`) / 3.2 × 10⁻⁵ (layer 8 `down_proj`), all below
  the 10⁻⁴ threshold. The replay reproduced the original values exactly (1.17 × 10⁻⁴, 1.08 × 10⁻⁴, 9.42 × 10⁻⁵) and
  attributes them to the reference: 1.17 × 10⁻⁴, 1.08 × 10⁻⁴ and 9.42 × 10⁻⁵ from the FP32 reference alone, against
  3.4 × 10⁻⁶, 1.9 × 10⁻⁶ and 1.4 × 10⁻⁶ from the update alone. The remaining error comes from the FP32 training bases
  (computed once in FP32, as prescribed) not being exactly the top-k singular vectors; it is largest where the update is
  large or σₖ and σₖ₊₁ nearly coincide.

## Attribution

- lm-evaluation-harness (EleutherAI): benchmark tasks and scoring; `t1_eval.py` subclasses its `HFLM`.
- Hugging Face `transformers` / `datasets`; models and data: Qwen3-0.6B, MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC.
- OPLoRA projections follow Xiong and Xie, *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during
  Parameter-Efficient Fine-Tuning* (Sections 3.2–3.3); LoRA follows Hu et al. (2022); L2-SP follows Li et al. (2018).
- `first_plot_questions.yaml` is copied unchanged from github.com/emergent-misalignment/emergent-misalignment
  (`evaluation/first_plot_questions.yaml`).
- Parts of this code were written with LLM assistance (Anthropic's Claude), as the course permits.
