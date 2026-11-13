# PA3 · Task 3 · Test-time compute: sampling, selection and adaptive inference

Uses the supplied 32 rollouts per question (Qwen2.5-3B-Instruct on 1,000 MATH test problems; Qwen2.5-VL-3B-Instruct on
the 1,000 MathVista testmini questions) to study repeated sampling, confidence- and vision-based selection, answer
consensus and adaptive stopping offline, plus one live experiment on 50 MATH questions comparing a single longer
response, parallel sampling and sequential refinement under the same 3,072-token output allowance.

## Files

| File | What it does |
|---|---|
| `t3_data.py` | loads and joins the supplied parquet files, checks alignment and the README totals, splits, random orders, answer grouping and grading (math-verify; MathVista's official rules) |
| `t3_methods.py` | selection rules on a visible prefix: FirstAns, self-consistency, highest score, rank-weighted consensus (RWC), VisConf calibration (rollout-level and tokenwise) |
| `t3_offline.py` | Steps 1, 2, 3, 5 (and the data for 6): every rule at N = 1…32 over ten orders, coverage, sampling-vs-selection decomposition, paired bootstrap, signal analysis, stopping controllers |
| `t3_live.py` | Step 4: the live single / parallel / refinement comparison (resumable, timed, seeded) and its scoring |
| `t3_report.py` | tables and figures from the saved results only |
| `t3_extra.py` | failure analyses from the saved results only: what won the vote when self-consistency missed, a vote-without-NO_ANSWER diagnostic, MathVista answers the official rules cannot read, live revisions |
| `../notebooks/t3_offline.ipynb` | Kaggle notebook (CPU) for the offline experiments |
| `../notebooks/t3_live.ipynb` | Kaggle notebook (GPU T4) for the live experiment |
| `../tests/test_task3.py` | 36 offline CPU checks on synthetic data shaped like the supplied files, plus a tiny random model for the live code |

## Reproducing (Kaggle)

1. Upload the supplied `ATML-PA3-Data` folder, unchanged, as a Kaggle dataset (here `atml-pa3-data`), and this package as `pa3-task3`.
2. `notebooks/t3_offline.ipynb`: Accelerator None (CPU), Internet on, attach both. About 20–40 min. Download `t3_offline_results.zip`.
3. `notebooks/t3_live.ipynb`: GPU T4, Internet on, attach both. About 1.5–4 h; every call is saved at once, and a rerun with
   the earlier output attached continues where it stopped. Download `t3_live_results.zip`.
4. Unzip both into `results/task3/` (they create `results/task3/t3/`). Tables and figures can be rebuilt without the data:
   `python -m task3.t3_report --results results/task3/t3 --out results/task3/t3_report_rebuilt`

Extra analyses (no data, no GPU): `python -m task3.t3_extra --results results/task3/t3 --out results/task3/t3_extra`

## Results layout (`results/task3/`)

| Folder | Contents |
|---|---|
| `t3/` | the original files from the two notebook outputs (`t3_offline_results.zip`: `prep/`, `analysis/`, `report/`; `t3_live_results.zip`: `live/`), unchanged |
| `t3_report_rebuilt/` | tables and figures rebuilt from `t3/` with `t3_report.py` (adds the live summary to the offline report) |
| `t3_extra/` | tables computed from `t3/` by `t3_extra.py` |

Offline tests (CPU, no downloads): `python tests/make_tiny.py /tmp/tinyqwen && python tests/test_task3.py /tmp/tinyqwen`

## Settings (as prescribed by the manual and the supplied README)

Splits: question IDs sorted lexicographically, permuted with `default_rng(6304)`; first 200 development, remaining 800
held-out; the first 50 held-out MATH questions for the live experiment. Budgets N ∈ {1, 2, 4, 8, 16, 32}; ten seeded
random orders of the 32 rollout IDs per question with nested prefixes, reused by every rule and stopping policy; outcomes
averaged over orders within a question, then over questions. Ties: earliest supporting rollout in the current order;
average ranks for tied RWC scores. Token cost = retained output tokens of every candidate a rule consumes (FirstAns: its
first candidate only). 95% intervals: 1,000 paired bootstrap resamples of questions (seed 6304), after averaging over orders.
Scores: Lᵢ = mean `logp`; Kᵢ = mean `kl_u_p`; vᵢ,ₜ = image / (image + prompt-text + generated-text attention); VisConf
calibrates the pooled token KLs of the visible candidates with their 5th and 95th percentiles (linear interpolation),
clips to [0, 1], averages per rollout (Cᵢ) and multiplies by Vᵢ; tokenwise fusion averages cᵢ,ₜ vᵢ,ₜ. RWC weight
(N − rᵢ) / (N − 1). Stopping checks after 4, 8, 16 candidates, otherwise 32; τ ∈ {0.5, 0.625, 0.75, 0.875, 1.0},
γ ∈ {0, 0.1, 0.25, 0.5}; the RWC quality score is self-certainty for MATH and VisConf for MathVista; per controller family,
the cheapest setting within 1 percentage point of fixed-32 self-consistency on development (ties: higher accuracy, smaller
τ, smaller γ; always-32 as the fallback), frozen before held-out evaluation.
Live: `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, FP16, no quantization, batch size 1,
temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0, native chat template with system message "You are a helpful
assistant.", the problem, two newlines, then the prescribed instruction; answers from the last balanced `\boxed{}`.

## Decisions where the manual leaves a choice

| # | Decision | Choice |
|---|---|---|
| D-T3-1 | Scorer | MATH: `math-verify` 0.9.0, each answer wrapped as `\boxed{…}` so LaTeX is parsed. MathVista: a port of the official `normalize_extracted_answer` (option letter, most-similar option, `int(float())`, `round(float(), precision)`, `str()` for lists) with the released testmini metadata from `AI4Math/MathVista` (revision recorded in `prep/scorer.json`); multiple-choice answers as option letters; the gold answer goes through the same rules |
| D-T3-2 | Answer groups (no gold) | each new answer is compared with the first answer of every existing group, in rollout-ID order; MATH: math-verify equivalence in either direction; MathVista: equal normalised strings |
| D-T3-3 | Missing answers | a null or empty extraction, or a MathVista answer the official rules cannot normalise, joins the single `NO_ANSWER` group (always incorrect, kept in votes and costs) |
| D-T3-4 | What a rule returns | one rollout: the highest-scoring one, or the earliest rollout (in the current order) of the winning answer group; its own answer is graded |
| D-T3-5 | Random orders | one `default_rng(6304)` per dataset, questions visited in sorted-ID order, ten permutations each (saved in `prep/*_orders.npz`) |
| D-T3-6 | Reporting split | every reported number uses the 800 held-out questions; the 200 development questions only choose stopping thresholds |
| D-T3-7 | Step 3 associations | within-question Spearman correlation with correctness (questions with both correct and incorrect rollouts), averaged over questions; top-vs-bottom quartile correct rates with 25th/75th percentile cut-offs (ties included), averaged over all and over mixed questions |
| D-T3-8 | Confidence–vision correlation | Pearson correlation across tokens within each rollout (at least 3 tokens, both series non-constant), with c calibrated on the question's 32 rollouts; per question the median over its rollouts; reported as median, quartiles and share positive across questions |
| D-T3-9 | Live run | all three strategies on one GPU (comparable runtimes); seed 6304 + 10 × question index + call index; the stop token is not counted as output; truncated = reached the cap without a stop token; one warm-up generation excluded from timing |
| D-T3-10 | Example selection | the first held-out questions (split order) showing each behaviour: self-consistency vs the main quality rule (SelfCert+RWC for MATH, VisConf+RWC for MathVista) at N = 32, order 0; a wrong early stop by the chosen quality-aware controller, order 0; a revision that changed the answer. Signal trajectories: the first 3 held-out MathVista questions with a correct and an incorrect rollout of at least 32 tokens (the smallest such rollout IDs) |
| D-T3-11 | Edge cases | a rollout with no tokens has undefined scores and ranks last; a token whose three attention masses are all zero has v = 0 |

## Implementation notes

- The loader never relies on row order: every file is joined to `generations.parquet` on (dataset, sample_id, rollout_id),
  and the joined lists are checked for equal lengths (token IDs, `logp`, `kl_u_p`, the attention lists), the attention
  `n_generated_text_tokens` lists are checked to be 0…T−1, and the totals in the supplied README are reproduced.
- VisConf calibration is recomputed for every visible prefix and never uses unrevealed rollouts (tested by changing an
  unrevealed rollout's tokens).
- Offline token cost is a proxy: it excludes prefill, attention collection and scoring.

## Attribution

- Kang et al., *Scalable Best-of-N Selection for Large Language Models via Self-Certainty*; Usman et al., *VisConf:
  Quality-Aware Consensus for Best-of-N in Vision-Language Model Reasoning* (calibration, fusion, RWC); Wang et al.,
  *Self-Consistency Improves Chain of Thought Reasoning in Language Models*; Madaan et al., *Self-Refine*.
- `math-verify` (Hugging Face) for MATH answer equivalence. `mathvista_normalize` ports MathVista's evaluation code
  (`evaluation/calculate_score.py`, Lu et al., *MathVista*). Data: the supplied ATML-PA3-Data rollouts; MATH
  (Hendrycks et al.) via `DigitalLearningGmbH/MATH-lighteval`; MathVista testmini metadata from `AI4Math/MathVista`.
- Parts of this code were written with LLM assistance (Anthropic's Claude), as the course permits.
