# Regeneration log

What was regenerated from the seed and compared with the saved results (T44, T46). The comparison is made by `python run.py check-regeneration`: tables must be exactly equal outside their timing columns, and models and the dataset are compared by hash.

## 1. Fresh clone (9 October 2026)

A clone of the repository from GitHub at commit `e749021`, with no `data/` and no `runs/` folder, on the project's laptop.

| Step | Result |
|---|---|
| `python run.py test` | 400 tests pass |
| `python run.py generate --width 5 --depth 4 --door W --items sofa,tv_unit,coffee_table` | works from the committed checkpoints: 62 of 64 candidates valid, 1.9 s, top 3 with quality 0.91, 0.91, 0.90 |
| `python run.py figures` | 21 figures redrawn from the saved tables and logs; `git status` shows no change, so they are byte-identical |
| `python run.py data` | dataset v1 rebuilt in about 5 minutes; content hash `ef1535ee101e7e8001c0098ce1c5d2e432686933f9dfbd05a7c20f36a7878f83`, identical to the recorded one |
| `python run.py check-regeneration` | all 20 tables equal, the dataset identical, the 4 committed checkpoints match their hashes: "everything compared is reproduced exactly" |

## 2. Checks made while building

| What | Result |
|---|---|
| The three frozen CVAE seeds, retrained from `configs/frozen.yaml` | bit-identical to the three MAE runs of the Gate 2 comparison (seed 0 also to the screening run trained three days earlier) |
| B1, B2 and G0 in E1 | their rows equal the Week 1 table `baselines.csv` in every column that is not a timing |
| The screening tables E2 to E7, rebuilt from the saved runs | every column they share with the saved tables is identical; each run's M1 sampling check, repeated on the GPU, gives exactly the stored raw valid rate |
| M2 with zero optimization steps | equals M1 (same draws), tested |

One thing does *not* carry over between devices: the same seed gives different random draws on the CPU and on the GPU, so a sampling check must be repeated on the device it first ran on.

## 3. Full regeneration into an empty folder

Started on 9 October 2026, after the checks above: every step of `python run.py all`, written to an empty folder (`SPACEGEN_OUTPUT`). This section is updated with the outcome.

## 4. What cannot be reproduced

Timings. The laptop's speed changes with its power and thermal state: the same baselines ran twice as fast one hour apart. Timing columns are left out of every comparison, and within a table all methods are timed in one pass, taking turns room by room.
