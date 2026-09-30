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

## Reproducibility

- `spacegen.seed.set_seed(seed)` seeds Python, NumPy and PyTorch and turns on PyTorch's
  deterministic mode, so two runs with the same seed on the same machine give bit-identical
  weights (tested for CPU and CUDA in `tests/test_seed.py`).
- Import `spacegen` before doing any CUDA work. Importing it sets `CUBLAS_WORKSPACE_CONFIG`,
  which only takes effect if it is set before the first GPU matrix operation; `set_seed`
  raises an error if CUDA was used first.
- Generated data (`data/`) and training runs (`runs/`) are not committed; they are rebuilt
  from the seed and the configs in `configs/`.

## Status

- [x] T01 Repository, seed helper, machine check
- [x] T02 Geometry
- [ ] T03 Catalog
