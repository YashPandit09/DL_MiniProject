# SpaceGen AI

Constraint-aware furniture layout generation with deep learning (Deep Learning mini project).
A conditional VAE proposes layouts for a room, latent optimization repairs constraint
violations, a rule checker verifies every layout, and a CNN evaluator ranks the survivors.

## Documents

| File | Contents |
|---|---|
| [PRD.md](PRD.md) | Goals, scope, user stories, success metrics, risks |
| [Tech_Spec.md](Tech_Spec.md) | Data format, rules, models, experiments, math appendix |
| [Architecture.md](Architecture.md) | Modules, data flow, design decisions |
| [Development_Plan.md](Development_Plan.md) | Tasks T01 to T50, gates, cut list, viva preparation |
| [reports/gate1.md](reports/gate1.md) | Gate 1 review: checklist evidence, key numbers, cross-teaching sheet, hours log |

## Setup

Python 3.11 or newer (tested on 3.12.5); the pinned NumPy and SciPy need 3.11+.

On the GPU laptop, install the CUDA build of PyTorch first, then the rest:

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

On a CPU-only laptop, `pip install -r requirements.txt` is enough. Then check each machine:

```
python run.py check-env
```

It prints the versions, the GPU and its memory, whether two seeded training runs give identical
weights, and how long one evaluator training epoch takes.

## Tasks

Windows has no `make`, so every step runs through `run.py`. Where `make` exists, `make <task>`
does the same thing.

| Command | What it does |
|---|---|
| `python run.py test` | Run the unit tests (extra arguments go to pytest, for example `python run.py test -k geometry`) |
| `python run.py check-env` | Versions, GPU and memory, determinism check, evaluator training speed |
| `python run.py generator-report` | Generate 2,000 Set A layouts, print acceptance, styles and quality, and plot the share of attempts rejected by room area and item count (`reports/figures/generator_rejection.png`; `--layouts N` and `--seed S` change the run) |
| `python run.py set-b-report` | Generate 2,000 Set B layouts and print the share valid and mean quality per perturbation type |
| `python run.py data` | Build the dataset in `data/v1/` (about 7 minutes): Set A (30k), Set B (60k), splits, held-out sets, diversity reference, f_max calibration, `metadata.json` with a content hash, and the figures `reports/figures/dataset_v1_*.png` (`--figures-only` redraws them) |
| `python run.py baselines` | Evaluate B1, B2 and G0 on 500 test rooms of dataset v1, 64 samples each (about 4 minutes); writes `reports/tables/baselines.csv` |
| `python run.py train-evaluator` | Train the CNN evaluator on Set B on the GPU (one epoch by default; `--epochs N --name RUN`); writes `runs/evaluator/RUN/` with `run.json` (seed, configs, git commit, dataset hash), `log.csv` and `model.pt` |
| `python run.py train-cvae` | Train the CVAE on Set A (about 1.5 minutes on the GPU): KL annealing, early stopping, per-epoch log with gradient norms, dead and active units, and a first M1 check; `--name RUN --set cvae.<field>=<value>` for the experiments |
| `python run.py evaluator-report` | E9a metrics of the evaluator in `runs/evaluator/e9a/` on the Set B test split: `reports/tables/evaluator.csv`, `evaluator_per_type.csv`, `reports/figures/evaluator_confusion.png` |
| `python -m spacegen.viz layout.json layout.png` | Draw a layout JSON as a floor plan with its hard-check results |

## Reproducibility

- `spacegen.seed.set_seed(seed)` seeds Python, NumPy and PyTorch and turns on PyTorch's
  deterministic mode, so two runs with the same seed on the same machine give bit-identical
  weights (tested for CPU and CUDA in `tests/test_seed.py`).
- Import `spacegen` before doing any CUDA work. Importing it sets `CUBLAS_WORKSPACE_CONFIG`,
  which only takes effect if it is set before the first GPU matrix operation; `set_seed`
  raises an error if CUDA was used first.
- Generated data (`data/`) and training runs (`runs/`) are not committed; they are rebuilt
  from the seed and the configs in `configs/`.
  `data/v1/metadata.json` records the seed, the configs, the git commit and a SHA-256 hash of
  the data contents; a rebuild on the same machine gives the same hash.

## Status

- [x] T01 Repository, seed helper, machine check
- [x] T02 Geometry
- [x] T03 Catalog (living room; the bedroom is P2)
- [x] T04 Rules file: every threshold in `configs/rules.yaml`, each marked as an assumption
- [x] T05 Door geometry and hard checks H1 to H3
- [x] T06 2D floor plans and the layout JSON format
- [x] T07 Evaluator input: 4-channel fractional-coverage rasters, built on the GPU
- [x] T08 Reachability (H4)
- [x] T09 Quality score (alignment, relations, circulation, space)
- [x] T10 Layout vectors (condition c: 25 values, target x: 36), canonical form, dataset tensors on the GPU
- [x] T11 Set A generator with all three living-room styles: (a) wall sofa, (b) floating sofa,
  (c) L-shape (styles (b) and (c) moved here from T13, which keeps the Set B perturbations);
  furniture that grows with room area; rejection report
- [x] T13 Set B for the evaluator: half generator layouts, half perturbed (jitter, rotation, random,
  forced overlap, near-miss at the overlap tolerance and at the door zone), each labelled by the checker
- [x] T14 Splits 70/15/15; held-out rooms (interpolation 22 to 26 m², above 32 m², out of range) that
  training never sees; diversity reference of generator layouts
- [x] T12 Baselines: B1 uniform random, B2 statistical sampler fitted on the Set A training split,
  G0 the generator
- [x] T15 Dataset v1: 30k Set A, 60k Set B (62% valid), held-out sets, metadata and content hash,
  figures; f_max calibrated to 0.38 (the spec's start was 0.45)
- [x] T16 Evaluation harness (`spacegen/metrics.py`, `spacegen/evaluate.py`) and the baseline table:
  raw valid B1 10%, B2 58%, G0 75% per attempt
- [x] T17 Notebook `notebooks/01_manual_backprop.ipynb`: hand-derived gradients of a two-item toy
  network (masked MSE, cross-entropy, penetration-depth overlap) equal autograd to 2e-15
- [ ] T18 Gate 1 review: 7 of 9 boxes done with evidence in `reports/gate1.md`; the cross-teaching
  session and the Week 1 hours log are for the team
- [x] T19 CVAE (`spacegen/models/cvae.py`): encoder, decoder with a logit rotation head, masked loss
  and KL, every Week 2 experiment switch in `configs/default.yaml`; 177,732 parameters; memorizes 64
  layouts within 500 steps
- [x] T20 CNN evaluator (`spacegen/models/evaluator.py`, `spacegen/train_evaluator.py`): 585,682
  parameters, Set B rasterized on the GPU, 19 s per epoch; validation accuracy 94% after 10 epochs
- [x] T21 CVAE training (`spacegen/train_cvae.py`): default run stops at epoch 119 (checkpoint 104), 8 of 16
  latent units active, no dead units; first M1 check 12% raw valid (before tuning and latent optimization)
- [x] T22 Evaluator, E9a (30 epochs, best epoch 26): test accuracy 96.6%, F1 0.973, ROC-AUC 0.991; hardest
  type near-miss overlap (accuracy 81%); Spearman with the rule score 0.66
