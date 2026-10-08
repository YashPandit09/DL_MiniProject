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

**Not run yet.** It needs four to eight hours on mains power, and the laptop was on battery when everything else was finished (9 October 2026, 00:50). Sections 1 and 2 are the evidence so far: the dataset, the final models, the figures and every table that was recomputed are reproduced exactly.

To run it (PowerShell; any empty folder will do):

```
$env:SPACEGEN_OUTPUT = "C:\spacegen-regeneration"
python run.py all                  # about eight hours in one go; --from STEP --to STEP runs a part
python run.py check-regeneration   # must end with "everything compared is reproduced exactly"
```

`python run.py all --list` prints the eighteen steps. The comparison ignores timing columns, so independent steps may also be run side by side to save time: after `frozen-seeds`, the steps `e1`, `e8`, the three `e10-...` steps, `e12` and `failures` do not depend on one another (run `e10` after `e1` and the three `e10-...` steps, then `figures`).

When it has passed, add its date and the last line of the check here, tick the last boxes in the Development Plan (Section 7, Final) and create the tag: `git tag final && git push origin final`.

## 4. What cannot be reproduced

Timings. The laptop's speed changes with its power and thermal state: the same baselines ran twice as fast one hour apart. Timing columns are left out of every comparison, and within a table all methods are timed in one pass, taking turns room by room.
