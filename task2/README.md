# PA3 · Task 2 · Vision-language model and visual grounding

Builds a SigLIP → MLP connector → Qwen3-0.6B vision-language model, trains it through a three-stage curriculum
(caption alignment, visual instruction tuning with replay, natural VQA), and tests whether its answers depend on the
image using complementary pairs under four image conditions (correct, complementary swap, mismatched, no image).

## Files

| File | What it does |
|---|---|
| `t2_data.py` | selects every image / question / pair with seed 6304 (COCO 2014 captions + instances, VQAv2 complementary pairs); official VQA answer normalisation and soft accuracy |
| `t2_features.py` | fetches only the needed COCO images (or uses attached COCO 2014/2017 folders); caches frozen SigLIP patch features (196 × 768, FP16); 2 × 2 pooling for 49 tokens |
| `t2_model.py` | connector, prompt / label / position construction, the VLM forward pass, per-example NLL, batched greedy decoding, calibration |
| `t2_train.py` | the curriculum (one process per run); development evaluation; checkpoints; pause / resume |
| `t2_eval.py` | final evaluation: both test suites × four image conditions, pair metrics, 4,000-resample pair bootstrap |
| `t2_report.py` | learning curves, result tables, correct-image advantages, aligned examples (stated rule), resources, 196 vs 49 |
| `t2_extra.py` | analyses from the saved evaluation files only: what answers do under the swap, agreement with no-image answers, paired 196 vs 49 bootstrap |
| `t2_preflight.py` | real-model checks before training (prompt layout, frozen sets, connector gradients, calibration, LoRA size) |
| `../notebooks/t2_vlm.ipynb` | Kaggle notebook running everything |
| `../tests/test_task2.py` | 71 offline CPU checks on synthetic COCO / VQAv2 data with tiny random Qwen3 and SigLIP |

## Reproducing (Kaggle, GPU T4 ×2, Internet on)

1. Upload this package as a Kaggle dataset named `pa3-task2` (or set `REPO_URL` in cell S1).
2. Run `notebooks/t2_vlm.ipynb` with **Save Version → Save & Run All**, attaching `pa3-task2`. About 4–5 h.
3. Download `t2_results.zip` from the output and unzip it into `results/task2/` (it creates `results/task2/t2/`).
4. Extra analyses (no GPU): `python -m task2.t2_extra --results results/task2/t2 --out results/task2/t2_extra`

Checkpoints every 250 updates (plus one whenever an hour passes without a save); anything still training after 11 h saves
and pauses so the output is kept; running again with that output attached reuses the selection, features and finished
runs, and resumes paused runs exactly (tested: pause + resume reproduces an uninterrupted run bit for bit).

## Results layout (`results/task2/`)

| Folder | Contents |
|---|---|
| `t2/` | the original files from the notebook output (`t2_results.zip`), unchanged: selection IDs, feature index and cache stats, training and development logs, per-question predictions for every condition, final metrics, report tables and figures |
| `t2_extra/` | tables computed from the files in `t2/` by `t2_extra.py` (step 4 above) |

Offline tests (CPU, no downloads):
`python tests/make_tiny.py /tmp/tinyqwen && python tests/make_tiny.py /tmp/tinysiglip --siglip && python tests/test_task2.py /tmp/tinyqwen /tmp/tinysiglip`

## Settings (as prescribed by the manual)

google/siglip-base-patch16-224 (frozen; 196 final patch features in 14 × 14 order, final LayerNorm applied exactly once,
cached in FP16, deterministic preprocessing) → connector Linear 768→1024, GELU, Linear 1024→1024 (trainable throughout) →
Qwen/Qwen3-0.6B (original weights frozen; LoRA r 16, α 32, dropout 0.05, no biases, on q/k/v/o in stages 2–3),
`enable_thinking=False`, seed 6304. Microbatch 8 × accumulation 2 = 16; AdamW, weight decay 0.01; clipping 1.0; per-stage 5%
warm-up then cosine decay to 10% of peak; gradient checkpointing; FP16 autocast with dynamic loss scaling, frozen FP16
backbone, FP32 trainable parameters and optimizer states; content caps: question 80, caption 64, short answer 16 tokens.
Loss on caption / answer tokens and `<|im_end|>`, averaged within each example, then across the effective batch.
Stages: 2,250 updates (connector, LR 1e-3; two passes over 18,000 captions), 1,500 (connector 1e-4, LoRA 2e-5; draws
presence 50% / counting 25% / caption 25%), 750 (connector 5e-5, LoRA 1e-5; draws VQAv2 70% / presence 10% / counting 10% /
caption 10%); each draw is a complete pair or two captions. Development evaluation at 0, every 500 updates, 2,250, 3,750 and
4,500. Final evaluation: greedy, 16 new tokens, batch size 8; grounding scored by normalised exact match, VQAv2 by local soft
consensus accuracy; 95% intervals from 4,000 bootstrap resamples of complete pairs (matched across conditions).

## Decisions where the manual leaves a choice

| # | Decision | Choice |
|---|---|---|
| D-T2-1 | Prompt wording | caption: "Write a short caption for this image."; questions: the question followed by "Answer the question using a single word or phrase." (LLaVA-1.5's short-answer prompt) |
| D-T2-2 | Label filters for generated questions | presence: the "yes" image has an instance covering ≥ 1% of the image, the "no" image has no label of that category at all; counting: every instance of the category ≥ 1% of the image and no crowd labels; counts 1–4 |
| D-T2-3 | Grounding training pools and sampling | pools built from the 6,000 caption-training images (images recur across stages); category-balanced draws (a category uniformly, then one of its pairs) |
| D-T2-4 | Captions per image | the 3 (training) or 1 (development) captions with the lowest annotation IDs |
| D-T2-5 | Connector calibration | fixed scale s = mean ‖text embedding‖ ÷ mean ‖fresh connector output‖, from training captions/questions and 512 training images; recorded in `calibration.json` |
| D-T2-6 | Presence / counting split | development 12 + 12 pairs, test 32 + 32 pairs |
| D-T2-7 | Optional 49-token run | done, on the second GPU in parallel; new connector and adapters, same seed, data, budgets and evaluation |
| D-T2-8 | Image source | only the needed COCO images are downloaded from images.cocodataset.org; attached COCO 2014 or 2017 folders are used when present (same image IDs) |

## Implementation notes

- Each stage starts a fresh AdamW (its parameter set and learning rates change). Warm-up: update n of a stage uses
  peak × n ÷ W for n ≤ W = ⌈5% of the stage⌉; the stage's last update uses 10% of the peak.
- Prompt pieces are tokenised separately and concatenated, so content caps are exact and identical across examples.
- Visual positions hold Qwen's `<|image_pad|>` placeholder (its embedding is never used; it is replaced by the connector output).
- VQAv2: consensus = most common normalised answer (ties: first seen); agreement = its count; training target = most common
  original string within the consensus group. Pairs need different consensus answers, ≥ 8/10 agreement on both images,
  the same answer type and the same question text.
- Evaluation sets: each val2014 image is used by at most one evaluation item, so all sets are image-disjoint (checked).
- Mismatched images: pairs are permuted with the seeded generator and pair p takes the images of pair p + 1 (same slot),
  fixed in the selection file before any generation.
- Example-selection rule for the qualitative figure: per suite and question type, the first two pairs (saved order) whose
  two correct-image predictions differ, and the first two whose predictions are identical.
- SigLIP preprocessing uses the PIL backend (`SiglipImageProcessorPil` when available), so resizing does not depend on torchvision.

## Attribution

- Liu et al., *Visual Instruction Tuning* (LLaVA): connector design and the align → instruction-tune curriculum.
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP); Hugging Face `transformers`; Qwen3-0.6B.
- COCO 2014 captions and instances; VQAv2 and its complementary pairs (Goyal et al., *Making the V in VQA Matter*).
- `vqa_normalize` and `vqa_soft_accuracy` port the official VQA evaluation code (GT-Vision-Lab/VQA,
  `PythonEvaluationTools/vqaEvaluation/vqaEval.py`), including its punctuation, number-word, article and contraction rules.
- Parts of this code were written with LLM assistance (Anthropic's Claude), as the course permits.
