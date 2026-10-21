# Decoding-parameter benchmark (Ollama): temperature, top-k, top-p

How sampling settings affect **factual reliability, repeatability and structured output** in small local LLMs.
Tested here on `qwen2.5:1.5b` and `smollm2:1.7b` (Quadro P2000, 4 GB), with the harness also configured for the 3B-4B tier.

## Setup
```bash
pip install -r requirements.txt
ollama pull qwen2.5:1.5b && ollama pull smollm2:1.7b
python download_datasets.py --n 200 --seed 0     # TruthfulQA sample -> data/ (then HF_DATASETS_OFFLINE=1)
pytest -q                                        # 42 tests, no GPU or network needed
```
To change models, pass `--models ...` on every command, or edit `MODELS_MAIN` in `config.py`.

## Run order (reduced, ~1-2 h per experiment on a 4 GB GPU)
```bash
M="qwen2.5:1.5b smollm2:1.7b"
python runner.py run --smoke --controls                       # verify the pipeline first
python runner.py controls --models $M                         # does the sampler config reach the sampler?
python runner.py estimate --model smollm2:1.7b                # timing run -> projected hours
python runner.py plan --models $M                             # generations per experiment

python runner.py run --experiment determinism --models $M --det-reps 5 --det-seed-modes fixed varied
python runner.py run --experiment tqa_gen --models $M --n-questions 100 \
       --configs T0.0 T0.3 T0.7 T1.0 T1.3 T1.6 T1.0_k1 T1.0_k10 T1.0_p0.9 T1.3_p0.9
python runner.py run --experiment tqa_mc  --models $M --n-questions 100 --configs <same list>
python runner.py run --experiment fake    --models $M --configs T0.0 T0.7 T1.0 T1.6
python runner.py run --experiment json    --models $M
python analysis.py all                                        # tables -> results/tables, plots -> plots/
```
Every command is resumable: re-run it after a crash or Ctrl-C and finished generations are skipped.
`--backend mock --smoke` exercises the whole pipeline with no Ollama (mock numbers are meaningless).

### Flags that cut run time
| flag | effect |
|---|---|
| `--configs ...` | run only these sampling configs (keep each config's baseline in the set: T0.0 for the temperature sweep, T1.0 for top-k/top-p, T1.3 for the interaction configs) |
| `--n-questions N` | TruthfulQA questions (CIs widen: about +/-10 points at N=100) |
| `--det-reps N` | repeats per prompt for determinism (default 10) |
| `--det-seed-modes fixed varied unseeded` | which seed regimes to run |
| `--det-num-predict N` | cap output tokens for determinism (start a fresh results file if you change it) |
| `--templates plain careful` | prompt-template factor (doubles tqa_gen/fake) |

## Layout
| file | role |
|---|---|
| `config.py` | models, the 16 sampling configs, base options, templates |
| `download_datasets.py` | seeded, category-stratified TruthfulQA sample as JSONL + manifest |
| `data_builders.py` | determinism prompts (30), fake-entity probes, JSON tasks (procedural, seeded) |
| `experiments.py` | job planning, controls and their pass/fail evaluation |
| `runner.py` | resumable runner (`plan`, `estimate`, `run`, `controls`) |
| `backend.py` | Ollama client wrapper (+ mock) |
| `scorers.py` | all scoring (pure functions, tested) |
| `stats_utils.py`, `analysis.py` | cluster bootstrap, paired contrasts + Holm, GEE logit, plots |
| `judge.py` | optional fixed T=0 LLM judge for TruthfulQA |

## Design
Every call sets `temperature, top_k, top_p, min_p, repeat_penalty, num_ctx, num_predict` (and `seed` when used)
explicitly. Each sweep changes one knob and neutralises the others (`top_k=1000`, `top_p=1.0`); `ollama_default`
(T0.8, k40, p0.9, repeat_penalty 1.1) is the reference. Determinism uses three seed regimes (`fixed` = same seed each run,
`varied` = seed per rep, `unseeded`). Greedy configs are not repeated in accuracy experiments and are tested by the controls.

## Metrics
- **TruthfulQA generation**: truthful / false / abstain / unclear (token-F1 vs reference lists, heuristic). MC1 letter accuracy as a judge-free baseline.
- **Fabrication**: share of invented-entity prompts answered with details instead of declining; false-refusal rate on 10 real controls.
- **Determinism**: modal-output rate, unique ratio, mean pairwise word edit distance, first-divergence position, TARr/TARa (Atil et al.), 3-gram repetition.
- **JSON**: strict parse (`json.loads`), schema validity (strict and lenient), field accuracy, failure taxonomy, prompt-only vs `format=schema`.
- **Stats**: question-clustered bootstrap CIs, paired config-vs-baseline contrasts with Holm correction, GEE logistic regression, per-model collapse temperature.

## Status of the current results (tables from the reduced run)
Treat these as **preliminary**. Several cells are incomplete or uneven, so check `n_clusters` in every table before comparing configs.

| experiment | what the tables show | caveat |
|---|---|---|
| Controls | `varies_t16`, `k1_vs_t0`, `same_seed_repeat` pass 5/5 on both models, so options reach the sampler. `greedy_repeat` fails (qwen 2/5, smollm2 1/5) and `topk0_vs_neutral` fails 0/5 | `top_k=0` is not bit-identical to 1000; keep using 1000 as "off" |
| Determinism | seed regime matters far more than temperature or truncation: with `varied` seeds modal rate drops from 1.00 (T0.0, k1) to about 0.3 (T0.7+); with a `fixed` seed it stays about 0.92-0.98 even at T1.3 | `unseeded` rows exist only for a few prompts (n=5-22); about 15-40% of outputs hit the 160-token cap, which inflates divergence |
| TruthfulQA generation | truthful rate is low (about 0-24%) with wide CIs; only smollm2 T1.6 vs T0.0 survives Holm correction (p_holm 0.008) | configs have 10-100 questions each (runs incomplete); T1.0 rows are 25 or 10 questions, not 100 |
| TruthfulQA MC | qwen about 0.40-0.53, smollm2 about 0.10-0.30, no contrast survives correction | n=10-40 per config; chance is below 0.25 because options vary in number |
| Fabrication | 97-100% fabricated in every config, 0% false refusals | no variation to analyse: either the models never decline or the decline detector misses their phrasing (inspect raw outputs before reporting) |
| JSON | prompt-only validity falls with temperature (qwen 1.00 -> 0.90 -> 0.55; smollm2 1.00 -> 0.65 -> 0.45 strict); `format=schema` stays at 1.00 | only T0.0, T0.7, T1.0 were run; qwen failures are all markdown fences, smollm2 also has prose, unparseable output and wrong types |

