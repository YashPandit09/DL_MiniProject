# SpaceGen AI: Technical Specification

| Field | Value |
|---|---|
| Version | 1.3 (revised after three technical reviews; see change log in Section 12) |
| Companion docs | PRD.md, Architecture.md, Development_Plan.md |
| Scope | Living room (P0). Bedroom (P2). Same code, different config. |

> All numeric values marked **(start)** are starting points to be tuned. Values marked **(assumption)** are our own design choices and must be presented as such.

---

## 1. Conventions

- **Units:** meters internally. Angles are discrete (see rotation).
- **Room:** axis-aligned rectangle of width `W` (x-axis) and depth `D` (y-axis), origin at bottom-left, `x in [0, W]`, `y in [0, D]`.
- **Door:** on one wall (N, E, S, W), fixed door width `0.9 m` **(assumption)**. Walls run counter-clockwise and the *start corner* of each wall is: S wall `(0,0)` to `(W,0)`; E wall `(W,0)` to `(W,D)`; N wall `(W,D)` to `(0,D)`; W wall `(0,D)` to `(0,0)`. The door **centre** lies at distance `s = m_d + (L - 2*m_d) * o` from the start corner, where `L` is the wall length, `o in [0,1]` is the door offset and `m_d = 0.65 m` (half the door width plus a 0.2 m margin). The door therefore never touches a corner: its centre is at least 0.65 m and its edges at least 0.2 m from any corner.
- **Door clearance zone:** rectangle of door width by `0.9 m` **(assumption)** projecting into the room from the door. Furniture must not enter it.
- **Rotation class `r in {0,1,2,3}`:** the direction the item's front faces: 0 = North (+y), 1 = East (+x), 2 = South (-y), 3 = West (-x). Classes 1 and 3 swap the item's width and depth in the footprint.
- **Effective footprint:** `(w_eff, d_eff) = (w, d)` if `r` is even, else `(d, w)`.
- **Canonical labels:** some items look identical after certain rotations or swaps, which would give the network contradictory targets. The catalog stores `rot_symmetry` per item (1 = none, 2 = same after 180 degrees, 4 = same after any 90 degrees) and an optional interchangeable `group` of slots. The generator always writes the canonical form: `r` is reduced modulo `4 / rot_symmetry` (a coffee table only uses classes 0 and 1, a side table always 0), and items in an interchangeable group (the bedroom nightstands) are ordered along the bed's lateral axis (perpendicular to the direction the bed faces), so the order does not depend on jitter when the bed is against an east or west wall. The rotation loss is masked for `rot_symmetry = 4` items, and diversity ignores non-informative flips (a requested facing for a symmetric pinned item is compared modulo its symmetry). Because the stored rotation is arbitrary among equivalent faces, the checker, the raster and the quality score treat **all equivalent faces of a symmetric item alike** (Sections 3.3, 3.4 and 4.2).
- **Normalized coordinates:** `u = x / W`, `v = y / D`. The model predicts `(u, v)`; all geometry (overlap, rules) is computed in meters after `x = u*W`, `y = v*D`.

---

## 2. Data specification

### 2.1 Catalog (`configs/catalog.yaml`)
Each item: `id`, `slot`, `name`, `w`, `d`, `h` (m), `price` (INR), optional `variant`. Values below are **illustrative placeholders (assumption)**.

**Living room (K = 6 slots)**

| Slot | Item | Variants (w x d x h, m; price INR) |
|---|---|---|
| 0 | sofa | 3-seater 2.10x0.90x0.85, 28000; 2-seater 1.60x0.90x0.85, 19000 |
| 1 | tv_unit | standard 1.50x0.40x0.50, 9000; large 1.80x0.40x0.50, 12000 |
| 2 | coffee_table | standard 1.00x0.55x0.45, 6000 |
| 3 | bookshelf | standard 0.80x0.30x1.80, 7000 |
| 4 | armchair | standard 0.80x0.80x0.85, 8000 |
| 5 | side_table | standard 0.45x0.45x0.50, 2500 |

**Bedroom (K = 6 slots, P2):** 0 bed (queen 2.05x1.60 / single 2.00x1.00), 1 wardrobe (1.20x0.60), 2 nightstand_a (0.45x0.40), 3 nightstand_b (0.45x0.40), 4 desk (1.20x0.60), 5 desk_chair (0.50x0.50).

Sofa and tv_unit are mandatory for the living room; the rest are optional. The bed is mandatory for the bedroom.

Catalog flags: `coffee_table` has `rot_symmetry: 2`; `side_table` has `rot_symmetry: 4`; `nightstand_a` and `nightstand_b` form one interchangeable `group`; `tv_unit` has `needs_access: false` (a floor-standing unit placed against a wall that is viewed, not walked up to).

### 2.2 Room and door sampling ranges (assumption)
| Room | Width `W` | Depth `D` |
|---|---|---|
| Living room | 3.5 to 7.0 m | 3.0 to 6.0 m |
| Bedroom | 3.0 to 5.0 m | 3.0 to 5.0 m |

Door wall uniform over 4 walls; `o` uniform in `[0, 1]`. The centre formula in Section 1 keeps the door centre at least 0.65 m, and the door edges at least 0.2 m, from any corner.

### 2.3 Layout vector

For slot `k = 0..K-1`:

| Symbol | Meaning | Type |
|---|---|---|
| `m_k` | Presence mask (given by user selection) | binary, input |
| `(w_k, d_k)` | Item size from catalog | float, input |
| `(u_k, v_k)` | Normalized centre | float in (0,1), target |
| `r_k` | Rotation class | categorical (4), target |

**Condition vector `c`** (dimension `7 + 3K = 25` for K = 6):

| Part | Dim | Encoding |
|---|---|---|
| Room size | 2 | `W / 8`, `D / 8` (fixed reference scale 8 m) |
| Door wall | 4 | one-hot |
| Door offset | 1 | `o` |
| Presence masks | K | `m_k` |
| Item widths | K | `w_k / 3` (0 for absent) |
| Item depths | K | `d_k / 3` (0 for absent) |

**Target vector `x`** (dimension `6K = 36`): for each slot, `(u_k, v_k)` and one-hot `r_k` (4). Absent slots are zero-filled and masked out of the loss. Rotation targets are the canonical form defined in Section 1.

### 2.4 Dataset generation

**Set A (good layouts, trains the CVAE):** target 30,000 layouts per room type **(start)**.
1. Sample `W, D` and the door, then a furniture subset whose size depends on room area (small rooms get fewer optional items, so the data is not dominated by crowded rooms that get rejected) and variants (mandatory items always present).
2. Pick a **layout style** at random. Living-room examples: (a) sofa against a wall facing the TV unit on the opposite wall; (b) floating sofa facing a TV unit placed against the far wall; (c) L-shape with armchair perpendicular to the sofa.
3. Place items with rule-driven anchors (against walls, facing directions, distance ranges from `configs/rules.yaml`) plus Gaussian jitter (position sigma about 0.05 m; small variation in distances).
4. Run the hard checker; keep only layouts that pass. Cap attempts per sample. **Log the rejection rate by room-area bin and by item count** and plot it. Rejection sampling makes crowded small rooms rare in the data; state this in the report.

**Set B (labelled layouts, trains the evaluator):** target 60,000 layouts **(start)**, about 50% valid.
- Half from Set A style generation.
- Half perturbed: (i) position jitter with sigma in {0.1, 0.3, 0.6} m on a random subset of items; (ii) random rotation changes; (iii) fully uniform random placement; (iv) forced overlap by moving an item onto another; (v) near-miss: shift one item so the overlap area lands between 0.002 and 0.05 m² (around the checker tolerance) or a clearance is just met or just violated. Each Set B sample stores its perturbation type so results can be reported per type.
- Labels are computed by the checker: `valid in {0,1}` and `quality in [0,1]`.

**Outlier variant (for E2):** Set A plus 2 to 5% deliberately odd layouts (extreme positions), flagged in metadata.

**Splits:** 70 / 15 / 15 (train / validation / test), split by layout. Further test sets are **sampled separately** with the same generator, and any training room that falls in their regions is rejected, so training never sees them:
- **Interpolation set:** room areas in [22, 26] m² **(start)**. About 21% of rooms would fall here under uniform `W` and `D`, so this leaves a visible gap in the training distribution.
- **Unseen-combination set:** room areas above 32 m² **(start)**, about 12% of rooms. Each individual `W` and `D` value still lies inside the training range, so this tests unseen *combinations*, not true extrapolation.
- **Out-of-range set:** `W` in (7.0, 8.0] m and `D` in (6.0, 7.0] m, outside every training range (the 8 m raster canvas and the `W/8` scaling still fit). The app shows an out-of-distribution warning for such inputs.
- **Reference set for diversity:** for 200 test conditions, 20 procedural layouts each (G0), used for the diversity ratio in E1.

**Real-room set (P1):** 15 to 20 measured real rooms with furniture positions, never used for training.

**Storage:** `.npz` (arrays) plus `metadata.json` (generator config, seed, dataset hash). Rasters are produced on the fly.

---

## 3. Rule engine (`rules.py`)

### 3.1 Hard checks
| ID | Check | Definition | Tolerance |
|---|---|---|---|
| H1 | In-room | Each effective footprint lies inside `[0,W] x [0,D]` | 1e-3 m |
| H2 | No overlap | Pairwise intersection area of footprints | at most 0.005 m² **(start)** |
| H3 | Door clearance | Footprints do not intersect the door clearance zone | 0 area (1e-4 m²) |
| H4 | Reachability | Every item's access point reachable from the door (see 3.3) | n/a |
| H5 | Budget | Sum of selected catalog prices at most user budget | exact |

A layout is **valid** iff H1 to H4 hold (H5 is checked at selection time).

### 3.2 Overlap of two axis-aligned boxes
For boxes `i, j` with centres `(x, y)` and effective sizes `(w, d)`:

```
ox_ij = ReLU( min(x_i + w_i/2, x_j + w_j/2) - max(x_i - w_i/2, x_j - w_j/2) )
oy_ij = ReLU( min(y_i + d_i/2, y_j + d_j/2) - max(y_i - d_i/2, y_j - d_j/2) )
ov_ij = ox_ij * oy_ij            (area in m^2)
```

The exact area is what the **checker** uses. It is **not** suitable as a training or optimization loss: if one box's x-extent lies inside the other's (a side table beside a long sofa), `ox_ij` equals the smaller width and does not change when either box moves, so its gradient is 0 and a fully contained item gets no push at all.

For **latent optimization** use penetration depth instead, with clearance margin `mu = 0.05 m` **(start)**:

```
px_ij   = ReLU( (w_i + w_j)/2 + mu - |x_i - x_j| )
py_ij   = ReLU( (d_i + d_j)/2 + mu - |y_i - y_j| )
pen_ij  = min( px_ij, py_ij )            (0 if the boxes are separated on either axis)
L_ov    = sum_{i<j} pen_ij^2
```
`pen_ij` is the smallest shift that would separate the two boxes, so the gradient pushes the pair apart along the easier axis, even under full containment. Replace `|t|` by `sqrt(t^2 + 1e-6)` so the gradient is finite when two centres coincide. It is exactly 0 there by symmetry, but exact ties do not occur in practice; if one does, add a random offset of about 1e-3 m. The same formula is used between each item and the fixed door clearance rectangle.

### 3.3 Reachability (H4)
1. Grid of 0.10 m cells covering the room.
2. Mark cells covered by any furniture footprint as blocked.
3. Compute the Euclidean distance transform from blocked cells and from the walls. A cell is **passable** if its distance to the nearest obstacle or wall is at least `min_path_width / 2`, with `min_path_width = 0.6 m` **(assumption)**.
4. **Start cell** = the cell at the centre of the door clearance zone (0.45 m in front of the door centre); the door opening itself is treated as free space. Do not start from the cell just inside the door: it is closer than 0.3 m to the wall and would never be passable, so every layout would fail H4. Because H3 keeps furniture out of the zone, the start cell is passable in every layout that passes H3.
5. Label the connected components of the passable mask with `scipy.ndimage.label` (4-connectivity). The reachable region is the component that contains the start cell.
6. Each item has an **access point**: the point 0.35 m in front of the centre of its front face. The item is reachable if the access point's cell is in the reachable region, or lies within 0.3 m of a cell in it. For items with `rot_symmetry > 1` there is one access point on **every equivalent face** (two for `rot_symmetry = 2`, four for `rot_symmetry = 4`), and the item counts as reachable if **any** of them is, because the canonical rotation is arbitrary among equivalent faces. Otherwise a side table stored at `r = 0` and pushed against the north wall would fail H4, and a coffee table with a sofa in the narrow gap on its stored front side would fail although its other side is open. Items with `needs_access: false` in the catalog (for example the `tv_unit`, a floor-standing unit viewed from a distance) are skipped.
7. The **reachability ratio** = reachable items / items that need access; H4 requires 1.0.

### 3.4 Quality score `S in [0,1]` (soft)
```
S = w_a*S_align + w_r*S_rel + w_c*S_circ + w_s*S_space
```
Default weights `(0.30, 0.30, 0.25, 0.15)` **(assumption, configurable)**.

| Term | Definition |
|---|---|
| `S_align` | Fraction of items that should sit against a wall which are within 0.10 m of it and correctly oriented (orientation is checked only for items with `rot_symmetry = 1`; symmetric items are judged by position only) |
| `S_rel` | Mean of relation scores, for example: sofa faces the TV unit (angle error small) and their distance lies within the configured range; coffee table lies between them; bed headboard against a wall; nightstands beside the bed |
| `S_circ` | Fraction of free floor cells that are passable (walkability) |
| `S_space` | Peak score when furniture area / room area is inside a target band (for example 0.15 to 0.40), falling off linearly outside it **(assumption)** |

All distance and angle ranges live in `configs/rules.yaml` and are documented as our assumptions.

### 3.5 Feasibility pre-check
Reject early if `sum(footprint areas) > f_max * (room area - door clearance area)` with `f_max` calibrated from the generator's acceptance rate (start `0.45` **(assumption)**; this is too generous for the smallest rooms: all six items total about 4.12 m² against a 4.36 m² limit in a 10.5 m² room, yet most such layouts fail the 0.6 m path rule), or if the cheapest selection exceeds the budget. Return a message with a suggestion (remove an optional item or choose a smaller variant).

---

## 4. Models

### 4.1 CVAE generator

**Purpose:** learn `p(x | c)`, the distribution of good layouts given the condition.

**Dimensions (K = 6):** `x` = 36, `c` = 25, latent `z` = 16 **(start)**.

**Encoder** `q(z | x, c)`
```
input  [x ; c]           (61)
Linear(61, 256) -> BatchNorm -> Act -> Dropout(0.1)
Linear(256, 256) -> BatchNorm -> Act -> Dropout(0.1)
Linear(256, 16) -> mu
Linear(256, 16) -> logvar
z = mu + exp(0.5*logvar) * eps,    eps ~ N(0, I)          (reparameterization)
```

**Decoder** `p(x | z, c)`
```
input  [z ; c]           (41)
Linear(41, 256) -> BatchNorm -> Act -> Dropout(0.1)
Linear(256, 256) -> BatchNorm -> Act -> Dropout(0.1)
head_pos : Linear(256, 2K) -> Sigmoid                       (u_k, v_k in (0,1))
head_rot : Linear(256, 4K) -> reshape (K,4)                 (logits; softmax only at inference)
```

`Act` defaults to ReLU; the final choice comes from experiment E3a (BatchNorm on, the deployed setting). The position head activation (Sigmoid vs Linear plus clamp) is tested in E7.

**Task to activation to loss mapping**

| Output | Task | Output activation | Loss |
|---|---|---|---|
| Position `(u, v)` | Bounded regression | Sigmoid (compare Linear + clamp) | MSE (default), MAE, Huber compared |
| Rotation `r` | 4-class classification | Softmax (inside the loss; the network outputs logits) | Categorical cross-entropy via `F.cross_entropy` on the logits |
| Latent | Regularization | none | KL divergence to N(0, I) |

**Training loss (masked with a Hadamard product by `m_k`)**
```
L_recon = sum_k m_k * [ ell_pos( (u_k,v_k), (u_k*,v_k*) ) + lambda_rot * CCE(r_k) ]
KL      = -0.5 * sum_j ( 1 + logvar_j - mu_j^2 - exp(logvar_j) )
L       = mean_batch( L_recon ) + beta * mean_batch( KL )
```
- `ell_pos` is a sum over the two coordinates. `lambda_rot = 1` **(start)**.
- **KL annealing:** `beta` rises linearly from 0 to `beta_target` over the first 20 epochs **(start)**. The validation loss used for **early stopping and checkpoint selection is always computed with `beta = beta_target`**, and the patience counter starts only after annealing ends. Otherwise the rising `beta` makes the validation loss climb and stops training before annealing finishes.
- **`beta_target`:** sweep {0.01, 0.1, 0.5, 1.0} **(start)**. Because reconstruction errors on normalized coordinates are small, a large `beta` risks posterior collapse. Interpret the reconstruction term as a Gaussian log-likelihood with variance `sigma_x^2`, which effectively sets `beta`.
- **Optional physics-informed term (experiment):** add `lambda_ov * sum_{i<j} pen_ij^2` (penetration depth, Section 3.2) computed from decoded positions and ground-truth rotation. Toggle in E2 or E5.
- **Collapse diagnostic:** count *active latent units*, dimensions whose KL contribution exceeds 0.01 nats averaged over the validation set. If none are active, the decoder ignores `z` (collapse).

**Optimization defaults:** Adam, `lr = 1e-3`, batch 256, up to 200 epochs, early stopping on the `beta_target` validation loss with patience 15 (counted from epoch 21), weight decay 0 **(start)**. Keep the whole dataset on the GPU as tensors and index mini-batches directly (no `DataLoader`), so an epoch takes a fraction of a second.

**Parameter count:** roughly 0.2 M **(check in code)**.

### 4.2 CNN evaluator

**Input raster:** fixed physical canvas of 8 m x 8 m at 128 x 128 px (0.0625 m per pixel), so scale is preserved across rooms. The room sits at the canvas origin. Each pixel stores the **exact fractional area coverage** of each shape (anti-aliased), not a 0/1 value, so sub-pixel positions matter and small overlaps change the values. **4 channels:**

| Channel | Content |
|---|---|
| 0 | Room mask (coverage of the room) |
| 1 | Furniture coverage, **summed over items and not clipped**, so an overlap shows up as values above single-item coverage |
| 2 | Front-face strip (thin band along each item's front edge, encodes orientation). For items with `rot_symmetry > 1` the strip is drawn on **all equivalent faces**, so the raster carries no arbitrary orientation for them |
| 3 | Door and door-clearance zone |

**Resolution caveat:** the checker tolerates 0.005 m² of overlap. That is about 1.3 pixels of area, but along a 0.9 m edge it is a strip roughly 6 mm wide, a tenth of a pixel. Fractional coverage makes such a strip visible in principle (it changes the pixel values by about 0.1), but it sits at the limit of what the CNN can learn. The checker stays the authority, and F1 is reported separately for near-miss samples (see Metrics).

**Architecture**
```
Conv(4,16,3,pad1)  -> BN -> ReLU -> MaxPool2    (16 x 64 x 64)
Conv(16,32,3,pad1) -> BN -> ReLU -> MaxPool2    (32 x 32 x 32)
Conv(32,64,3,pad1) -> BN -> ReLU -> MaxPool2    (64 x 16 x 16)
Conv(64,64,3,pad1) -> BN -> ReLU -> MaxPool2    (64 x 8 x 8)
Flatten (4096) -> Dropout(0.3) -> Linear(4096,128) -> ReLU
head_valid : Linear(128,1)        -> Sigmoid  (in loss: BCEWithLogits)
head_score : Linear(128,1)        -> Linear   (regression, target in [0,1])
```
About 0.59 M parameters.

**Implementation.** Do not precompute rasters: Set B at 4 x 128 x 128 in float32 would need about 16 GB. Rasterize on the fly on the GPU. All footprints are axis-aligned boxes, so the exact coverage of a box in a pixel is the outer product of two 1-D overlaps (its x-interval with each column, its y-interval with each row); a batch of 64 layouts takes a few milliseconds. The same computation is piecewise linear in the box positions, which makes the differentiable version needed for M3 cheap.

**Training time and hardware.** A reviewer measured evaluator training at 128 x 128, batch 256, on 42k samples: about 2,600 samples/s (16 s per epoch) on the RTX 3050 and about 300 samples/s (140 s per epoch) on the CPU, before the extra cost of rasterizing on the CPU. The evaluator therefore needs the GPU; on CPU, use a reduced setting (64 x 64 raster, fewer samples, about 12 epochs fit in 30 minutes). The CVAE, latent optimization, rules and app are small enough for CPU. Re-measure on your own machines on Day 1, and schedule the single GPU explicitly (Development_Plan Section 2).

**Loss:** `L_eval = BCE(valid) + lambda_s * Huber_or_MSE(score)`, `lambda_s = 1` **(start)**. The score loss is applied on all samples (score is defined for invalid layouts too).

**Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix on valid vs invalid; Spearman correlation of predicted score against rule score. **Report F1 separately for each Set B perturbation type** (jitter levels, rotation change, random placement, forced overlap, near-miss); an overall F1 alone hides the hard cases.

**Role of the evaluator (honest statement).** The checker's rules and the rule score are exact, so the evaluator is **not needed for correctness**, and it is **not a speed-up**: reachability for 64 candidates takes roughly 10 to 30 ms on a laptop CPU, about the same as rasterizing and scoring them with the CNN on the GPU, and both are tiny next to the 5 s latency budget. Its purpose is therefore: (1) a controlled study of a CNN on raw layout images (E9); (2) an experiment on whether a learned score can act as a differentiable surrogate for the non-differentiable rule terms (reachability, alignment) inside latent optimization (variant M3, experiment E8b, P1; it needs testing because the optimizer can exploit a surrogate); (3) a learned ranker for the top 3, reported next to the exact rule score and their correlation, with the rule score as the reference; (4) a starting point that could later be fine-tuned on human preference ratings, which the rule score cannot capture.

**Baseline:** MLP on hand-crafted features (total overlap, out-of-room area, door-clear flag, minimum pairwise distance, reachability ratio, wall-contact count, relation errors). *Honest caveat:* these features encode the very rules that create the labels, so the MLP is expected to do very well. The comparison shows what raw-pixel learning costs (the classic "manual feature engineering vs learned features" trade-off), not that the CNN wins.

### 4.3 Baselines

| ID | Method | Description |
|---|---|---|
| B1 | Uniform random | Each item's centre uniform in the room, rotation uniform over 4 classes |
| B2 | Statistical sampling | From Set A train data, fit per-slot empirical distributions of `(u, v, r)` conditioned on room type and door wall bins (histogram or KDE); sample items sequentially; no neural network |
| M1 | CVAE only | Sample `z ~ N(0, I)`, decode |
| M2 | CVAE + latent optimization | M1 followed by the procedure in 5.2 |
| M3 (P1) | CVAE + latent optimization + CNN surrogate | M2 with the extra term `- lambda_c * q_hat` (needs the differentiable raster) |
| G0 | Procedural generator (reference) | The Set A generator with rejection sampling. Not a learned model and valid by construction, so it is a reference and is **not part of the validity comparison** (goal G2 is limited to B1 and B2). Its raw valid rate is defined as its **acceptance rate per attempt**. Report its **cost per valid layout** (attempts and seconds) and its diversity (the reference for the diversity ratio) |
| G0-pin (P1) | Procedural generator with a pin | Run the generator, move the pinned item to the requested position, then filter with the checker (repeat until valid or a cap). The strongest non-neural baseline for E12 |

All methods report raw validity (no filtering) and yield after filtering, with identical numbers of samples per input.

---

## 5. Inference pipeline

### 5.1 Steps
1. Validate input; run feasibility pre-check.
2. Select catalog variants if the user left them open (rule-based selection under budget and footprint limits; not learned). If several combinations fit, pick the one with the largest total footprint that still fits (comfort proxy) **(assumption)**.
3. Build the condition vector `c`.
4. Sample `N = 64` latent vectors; decode to positions and rotation probabilities.
5. Set rotation = argmax (fixed thereafter), except that a pinned item with a requested facing gets that rotation.
6. Latent optimization (5.2), optional. Afterwards **snap each pinned item exactly onto its requested position** before any check runs.
7. Run hard checks H1 to H4 and discard failures. The checker is always the final authority.
8. Score survivors with the CNN evaluator (and compute rule quality score for display).
9. Select the top 3 (5.3).
10. Render and report metrics.

### 5.2 Latent optimization
The decoder is frozen and put in `eval()` mode (BatchNorm uses running statistics, dropout off). Only `z` is optimized, as a batch of 64 independent problems.

```
minimize over z (each candidate starts at z0 ~ N(0, I)):
L_c(z) = lambda_ov  * sum_{i<j} pen_ij^2              (penetration depth, Section 3.2)
       + lambda_rm  * sum_i out_i
       + lambda_dr  * sum_i pen(i, door_zone)^2
       + lambda_pin * sum_{k in P} || p_k - p_k* ||^2
       + lambda_z   * 0.5 * || z - z0 ||^2
       - lambda_c   * q_hat(z)                        (M3 only: CNN score)
```
- `out_i = ReLU(w_i/2 - x_i) + ReLU(x_i + w_i/2 - W) + ReLU(d_i/2 - y_i) + ReLU(y_i + d_i/2 - D)` (protrusion outside the room; uses effective sizes). A fully inside item has zero loss and zero gradient here, which is correct.
- The `lambda_z` term is an **anchor to the candidate's own starting point `z0`**, not a pull towards 0. `z0` is a draw from the prior, so it starts in a high-density region, and the anchor keeps each candidate near it. A `||z||^2` term would drag every candidate to the same point `z = 0` and destroy diversity.
- `P` is the set of pinned items (P1). A pin gives a requested centre `p_k*` in meters and, optionally, a facing (N, E, S, W). If a facing is given, the pinned item's rotation is **overwritten** with it before optimization; otherwise the decoder's argmax is kept, which could leave a pinned sofa facing the wall. After optimization each pinned item is **snapped exactly** onto its requested position, so the pin is honoured by construction; the `lambda_pin` term only makes the other items adapt to it while `z` moves.
- Optimizer: Adam, `lr = 0.05`, `T = 150` steps **(start)**; stop a candidate early when `L_c` is below a small threshold. Adam does not guarantee a monotone decrease, so success is judged by the final loss, not by every step.
- Rotation is fixed from the decoder's argmax (non-differentiable choice); only positions move.
- Weights `(lambda_ov, lambda_rm, lambda_dr, lambda_pin, lambda_z)` start at `(10, 10, 10, 20, 0.05)` **(start)**; tune on the validation set.

### 5.3 Top-3 selection with diversity
1. Sort valid candidates by evaluator score (descending).
2. Greedily add a candidate if its mean per-slot displacement from every already-chosen layout is at least `tau_div = 0.3 m` **(start)** (rotation mismatch counts as a fixed penalty of 0.5 m for that slot).
3. If fewer than 3 pass, relax `tau_div` by 25% up to two times, then return what exists with a note.

---

## 6. Experiments

**Protocol.** Vary **one factor at a time** around the default configuration (no full grids). Screen every setting with **one seed**, then re-run the default and the best one or two settings with **three seeds** and report mean and standard deviation. No significance claims are made from three seeds. This is roughly 65 short CVAE trainings for E2 to E7 instead of about 210 with full grids.

**Order of work.** The ablations E2 to E7 finish first. The configuration is then **frozen** (Gate 2, saved as `configs/frozen.yaml`), and the headline experiments E1, E8 and E10 are run on the frozen configuration. Earlier runs of E1 and E10 on the default configuration are only for debugging the pipeline.

| ID | Question | Variables | Metrics | Plot or table |
|---|---|---|---|---|
| E1 | Does the neural pipeline beat the baselines, and at what cost? | B1, B2, G0 (reference), M1, M2 (M3 if built) | RVR, overlap, reachability, quality, diversity ratio to G0, **cost per valid layout** (attempts and seconds), latency | Bar chart and main results table |
| E2 | Which position loss is best, and how do losses handle outliers? | MSE, MAE, Huber (delta in {0.01, 0.05, 0.1}); clean vs outlier dataset; with/without overlap penalty | Position error (median and mean), RVR | Training curves; analytic plot of loss and gradient vs error |
| E3a | Which hidden activation should the deployed model use? | Sigmoid, Tanh, ReLU, LeakyReLU(0.01), ELU, GELU; depth 2; **BatchNorm on** (the deployed setting) | Validation loss, RVR after decoding | Table; decides T1 |
| E3b | How do activations behave without normalization (vanishing gradients, dying units)? | The same six activations; hidden depth in {2, 6}; **BatchNorm off**; plus BatchNorm on for Sigmoid and ReLU at depth 6 as a reference | Validation loss, per-layer gradient norm (epochs 1, 10, 50), dead-unit percentage | Gradient-norm plot; dead-unit bar chart; BN-on vs BN-off comparison |
| E4 | Which optimizer converges best? | SGD, SGD + momentum (0.9), RMSProp, Adam; lr grid | Epochs to reach target loss, final loss | Convergence curves |
| E5 | KL weight and latent size trade-off | beta in {0.01, 0.1, 0.5, 1.0}; z-dim in {4, 16, 32} | Reconstruction error, RVR, diversity, active units | Trade-off plot |
| E6 | Regularization effect | BatchNorm on or off; dropout in {0, 0.1, 0.3}; early stopping vs fixed epochs | Train-validation gap, validation loss | Curves and table |
| E7 | Does Sigmoid saturation hurt near walls? | Sigmoid vs Linear + clamp position head | Position error for items with `u` or `v` below 0.05 or above 0.95 vs the rest | Grouped bar chart |
| E8 | How much does latent optimization help? | T in {0, 25, 50, 100, 200}; lambda settings | RVR, diversity, runtime | RVR vs T curve |
| E8b (P1) | Does a CNN surrogate help latent optimization? | M2 vs M3 (adds `- lambda_c * q_hat`) | RVR, quality, reachability, and signs of exploiting the surrogate (CNN score rises, rule score does not) | Table |
| E9a | Evaluator quality | CNN; class balance; perturbation types | Precision, recall, F1 overall and **per perturbation type**, AUC, confusion matrix, Spearman with the rule score | Tables and matrices |
| E9b (P1) | CNN vs feature MLP | CNN vs MLP on hand-crafted features | Same metrics | Comparison table |
| E10 | Generalization beyond the training distribution | In-distribution test vs interpolation vs unseen-combination vs out-of-range sets | RVR and quality for M1 and M2 | Grouped bars |
| E11 (P1) | Behaviour on real rooms | 15 to 20 measured rooms | Qualitative plus RVR | Figure gallery |
| E12 (P1, top priority of the P1 items) | Pinned furniture: the strongest case for the learned model | 1 or 2 pinned items; M2 vs **G0-pin** (generator, move the pinned item to its spot, filter), and vs B1 and B2 with the pinned item fixed and the rest sampled then filtered. **Results are split by which item is pinned** (sofa, TV unit, coffee table, side table, ...) | Validity after snapping, cost per valid layout, quality, and the pre-snap displacement of the pinned item (how far optimization left it from the requested spot). The pin itself is exact by construction, so "pin satisfaction" is not the metric | Table per pinned item |

**Why E3 has two parts.** The deployed model uses BatchNorm, so the activation is chosen with BatchNorm on (E3a), where ELU, GELU and LeakyReLU are tested in the setting they will actually run in. The BatchNorm-off study (E3b) is a diagnostic: BatchNorm keeps pre-activations near zero mean and unit variance, which hides both vanishing gradients and dying ReLU. The BN-on references test that expectation directly; if BN does mask the effects, that is itself a finding to explain in the viva.

**Failure-case analysis:** collect at least 10 failed or poor outputs across methods, classify the cause (overlap, door blocked, unreachable, collapse), and discuss.

---

## 7. Metric definitions

```
RVR            = (# raw samples passing H1..H4) / (# raw samples)
Mean overlap   = mean over samples of sum_{i<j} ov_ij     (m^2)
Reachability   = mean over samples of reachable_items / items_that_need_access
Diversity      = mean over valid pairs (a,b) for one input of
                 (1/|K|) * sum_{k in K} ( ||p_k^a - p_k^b||_2 + 0.5*[r_k^a != r_k^b] )
Precision      = TP / (TP + FP)      Recall = TP / (TP + FN)
F1             = 2 * Precision * Recall / (Precision + Recall)
Spearman       = Pearson correlation of rank(score_pred) and rank(score_rule)
Active units   = # { j : mean_val KL_j > 0.01 }
```
`K` is the set of slots present in both layouts. Layouts are compared **after canonicalization**: for interchangeable groups take the minimum over permutations, and count a rotation mismatch only for slots with `rot_symmetry = 1` (compare modulo the symmetry otherwise), so meaningless flips do not inflate diversity. **Diversity ratio** = Diversity(method) / Diversity(G0 reference set), both over valid layouts for the same condition. **Cost per valid layout** = (time spent on sampling, optimization and checking) / (number of valid layouts produced). For G0 the raw valid rate is its acceptance rate per attempt, and G0 is left out of the validity comparison. **Expect G0 to be cheaper per valid layout than M2** for requests without a pin, because M2 makes the same checker calls plus 150 decoder passes. The learned approach has to earn its place through pinned completion (E12), amortized sampling and differentiable repair, and we report the numbers whichever way they fall.

---

## 8. Math appendix (viva preparation)

Both team members should be able to derive these on paper.

**A. Forward pass shapes.** For batch `B`, input `X in R^{B x d_in}`: `H = act(X W + b)` with `W in R^{d_in x d_out}`, `b in R^{d_out}` (broadcast). Matrix product `X W` is valid iff columns of `X` equal rows of `W`.

**B. Sigmoid and BCE.** `sigma(z) = 1/(1+e^{-z})`, `sigma'(z) = sigma(z)(1-sigma(z)) <= 0.25`. With `L = -[y log p + (1-y) log(1-p)]`, `p = sigma(z)`: `dL/dz = p - y`. Saturation: for large `|z|`, `sigma'` is near 0, so gradients vanish (relevant to positions near 0 or 1).

**C. Softmax and categorical cross-entropy.** `p_i = e^{z_i} / sum_j e^{z_j}`, `L = -sum_i y_i log p_i`: `dL/dz_i = p_i - y_i`.

**D. MSE, MAE, Huber.** MSE `L = (yhat - y)^2`, gradient `2 (yhat - y)` grows with error (outlier-sensitive). MAE gradient is `sign(yhat - y)` (constant, not differentiable at 0). Huber with threshold `delta`: `L = 0.5 e^2` if `|e| <= delta`, else `delta (|e| - 0.5 delta)`; `dL/de = e` or `delta * sign(e)`.

**E. ReLU family.** `ReLU'(z) = 1` for `z>0`, else 0 (choose 0 at `z=0`, a subgradient). Dying ReLU: a unit with `z<0` for all inputs gets zero gradient. Leaky ReLU keeps slope `alpha`. ELU: `alpha (e^z - 1)` for `z<0`, derivative `alpha e^z`.

**F. Backpropagation (one layer).** For `a = act(z)`, `z = W h + b`: `dL/dW = delta h^T`, `dL/db = delta`, `dL/dh = W^T delta`, where `delta = dL/da * act'(z)`.

**G. Adam.** `m_t = b1 m_{t-1} + (1-b1) g_t`, `v_t = b2 v_{t-1} + (1-b2) g_t^2`, bias correction `mhat = m_t/(1-b1^t)`, `vhat = v_t/(1-b2^t)`, update `theta <- theta - lr * mhat / (sqrt(vhat) + eps)`.

**H. ELBO for the CVAE.** `log p(x|c) >= E_{q(z|x,c)}[log p(x|z,c)] - KL( q(z|x,c) || p(z) )`. Maximizing the right side is minimizing `L_recon + KL`; `beta` re-weights KL.

**I. KL between a diagonal Gaussian and N(0, I).** `KL = 0.5 * sum_j ( mu_j^2 + sigma_j^2 - 1 - log sigma_j^2 )`, same as the code form with `logvar = log sigma^2`.

**J. Reparameterization.** `z = mu + sigma * eps`, `eps ~ N(0,I)`. Then `dz/dmu = 1` and `dz/dsigma = eps`, so gradients flow through the sampling step to the encoder.

**K. Overlap: why area fails as a loss and penetration depth works.** The exact intersection width is `ox = ReLU(min(x_i + w_i/2, x_j + w_j/2) - max(x_i - w_i/2, x_j - w_j/2))`. If box `j` lies inside box `i` along x, the `min` and `max` both pick box `j`'s edges, so `ox = w_j` regardless of position and `d ox / d x = 0`. A fully contained item therefore receives no gradient. Penetration depth `px = ReLU((w_i + w_j)/2 + mu - |x_i - x_j|)` has `d px / d x_i = -sign(x_i - x_j)` wherever `px > 0`, which pushes the centres apart even under full containment. With `pen = min(px, py)` the push acts along the axis that separates the boxes fastest (the minimum translation). Where `pen = 0` (separated) the gradient is 0: nothing to fix.

**L. Latent optimization with an anchor.** Minimizing `L_c(z)` with `0.5 lambda_z ||z - z0||^2` is a proximal (trust-region) step: the constraint terms act as a negative log-likelihood and the anchor as a Gaussian prior centred on the starting point `z0`, which was itself drawn from `N(0, I)`. It is not the global `N(0, I)` prior, which would pull every candidate to the same point `z = 0`.

**M. Data scaling.** Dividing coordinates by room size is scalar multiplication per axis (`u = x/W`). It makes the target independent of absolute room scale and keeps inputs in a range that suits Sigmoid and gradient-based training.

**N. Hadamard masking.** The loss term `m_k * (...)` is element-wise multiplication by the presence mask; absent items contribute zero loss and zero gradient.

**O. BatchNorm and vanishing gradients.** BatchNorm rescales pre-activations to roughly zero mean and unit variance before the activation. For Sigmoid this keeps inputs where `sigma'` is near its 0.25 maximum, and for ReLU it keeps a share of units active. With BN on, the vanishing-gradient and dying-ReLU effects that E3 tries to show are weakened, so E3 runs without BN and uses a small BN-on reference.

---

## 9. Implementation details

### 9.1 Repository structure
```
spacegen/
  configs/            default.yaml, catalog.yaml, rules.yaml
  spacegen/
    geometry.py       boxes, rotation, overlap (NumPy and PyTorch versions)
    catalog.py        load and query catalog
    rules.py          hard checks, reachability, quality score
    generator.py      procedural dataset generator (Set A, Set B)
    raster.py         layout to 4-channel raster
    dataset.py        PyTorch datasets, normalization, splits
    baselines.py      B1, B2, G0, G0-pin
    models/cvae.py
    models/evaluator.py
    models/mlp_baseline.py
    train_cvae.py
    train_evaluator.py
    latent_opt.py
    pipeline.py       end-to-end inference, top-3 selection
    metrics.py
    viz.py            2D plots (3D optional)
  experiments/        run_e1.py ... run_e12.py, make_figures.py
  app/streamlit_app.py
  notebooks/          01_manual_backprop.ipynb (toy example)
  tests/
  reports/            figures/, tables/
  data/               generated datasets (gitignored, regenerate by seed)
  README.md, requirements.txt, Makefile
```

### 9.2 Configuration
YAML files with a single `seed` and per-experiment overrides. Every training run writes its config, seed, git commit and dataset hash next to the checkpoint.

**Seed helper (deterministic mode).** One helper sets the Python, NumPy and PyTorch seeds and enables `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark = False` and the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`. On the same machine and software versions, two runs with the same seed then give bit-identical weights (a reviewer confirmed this for a CVAE-like MLP and the 4-conv CNN with BatchNorm, cross-entropy, BCE and Adam). Final checks on the same machine are therefore **exact-equality** checks. Across machines, compare within a stated tolerance: RVR within 2 percentage points and F1 within 0.02 **(start)**. If an operation has no deterministic implementation PyTorch raises an error; handle that case explicitly and document it.

### 9.3 Testing plan
| Test | Purpose |
|---|---|
| Overlap: known boxes with analytic area; symmetry `ov_ij = ov_ji`; NumPy vs PyTorch agree | Correct geometry |
| Overlap loss gradient: finite-difference check vs autograd, **including a fully contained box (must have a non-zero gradient)**; for coincident centres assert only that the gradient is finite (it is exactly 0 there) | Correct differentiability |
| Rasterizer: the GPU outer-product rasterizer equals a brute-force supersampled reference; summed coverage matches the analytic area; symmetric items get strips on all equivalent faces | Correct CNN input |
| Generator: 100% of Set A "good" layouts pass the checker | Trustworthy data |
| KL: closed form equals `torch.distributions.kl_divergence` | Correct loss |
| Manual backprop: hand-derived gradients of a toy 2-item network equal autograd to 1e-6 | Viva evidence |
| Reachability: an empty room passes; a blocked door, a walled-off item and a door near a corner behave as expected; the start cell is passable | Correct H4 |
| Reachability of symmetric items: a side table against the north wall passes; a coffee table with a sofa in the gap on one face passes if the opposite face is open | Access checked on all equivalent faces |
| Pipeline smoke test: end-to-end run with tiny models returns 3 layouts | Integration |
| Determinism: two training runs with the same seed give identical weights on the same machine | Reproducibility claim is exact |
| Door placement: for all 4 walls and `o` in {0, 0.5, 1}, the door lies inside the wall, its centre is at least 0.65 m and both door edges at least 0.2 m from the corners | Unambiguous door definition |
| Canonicalization: every generated layout is in canonical form; swapping nightstands or rotating a coffee table by 180 degrees gives the same encoded target | No contradictory labels |
| Training loop: early stopping never triggers before annealing ends; selection loss uses `beta_target` | Correct KL annealing |
| Latent optimization: final constraint loss is lower than the initial loss for at least 90% of candidates | Sensible acceptance test (Adam is not monotone) |

### 9.4 Tooling
Python 3.10 or newer; PyTorch (CUDA build if it works, otherwise CPU); NumPy; SciPy (distance transform); scikit-learn (metrics); pandas; Matplotlib; Streamlit; PyYAML; pytest. Plotly for the optional 3D view. Random seeds set for Python, NumPy and PyTorch.

### 9.5 Streamlit app
Sidebar inputs: room type, `W`, `D`, door wall and offset, furniture selection and variants, budget, number of candidates, latent optimization on or off, optional pinned item position and facing. Main area: top-3 layouts with metrics; a "Compare methods" tab (B1, B2, G0, M1, M2 on the same input); a "Training and results" tab showing saved figures. Export buttons for JSON and PNG.

---

## 10. Limitations and ethics (to state in the report)

- **Synthetic data:** the model can only reproduce our rule-based notion of a good layout. It does not learn real designer taste. Design rules and score weights are assumptions.
- **Simplified geometry:** rectangular rooms, one door, four rotations, no windows or built-ins.
- **Illustrative catalog:** prices and sizes are placeholders, not retail data.
- **Small-scale evaluation:** three seeds and a small real-room set give descriptive, not statistical, evidence.
- **Learned vs rule-based:** the procedural generator already produces valid layouts, so the case for the CVAE is fast sampling, differentiable repair, and completion around pinned items, not validity by itself. The evaluator is not needed for correctness (the rule score is exact); its value is the CNN study and its possible use as a differentiable surrogate.
- **Rasterization resolution:** near-miss violations at the checker's tolerance are close to the raster's resolution limit.
- **Rejection-sampling bias:** crowded small rooms are under-represented in the data.
- **Ethics:** no personal data; do not present outputs as professional architectural or safety advice (for example, egress or accessibility regulations are not modelled).
- **Deployment:** local Streamlit for the demo. For real use, wrap inference in a REST API (for example FastAPI), version models and datasets, validate inputs, and monitor for out-of-distribution rooms (inputs outside the training ranges should trigger a warning).

---

## 11. Open technical decisions

| # | Decision | Default | How to decide |
|---|---|---|---|
| T1 | Hidden activation | ReLU | E3a (BatchNorm on, depth 2) |
| T2 | Position loss | MSE | E2 |
| T3 | Position head | Sigmoid | E7 |
| T4 | `beta_target` | 0.1 | E5 |
| T5 | Latent size | 16 | E5 |
| T6 | Optimization steps | 150 | E8 |
| T7 | Bedroom in scope | No (P2); pinned furniture has priority | Development_Plan cut list |

---

## 12. Change log

**v1.1 (after technical review)**
- Overlap loss changed from intersection area to penetration depth (area has zero gradient under containment). Appendix K rewritten; test added.
- Reachability start moved to the centre of the door clearance zone; `scipy.ndimage.label` replaces the hand-written search.
- Early stopping and checkpoint selection now use `beta_target` and start after KL annealing.
- Raster changed to 128 x 128 with exact fractional coverage; F1 reported per perturbation type; near-miss perturbations added.
- Canonical labels for symmetric items and interchangeable slots; diversity computed after canonicalization.
- Door offset defined precisely (wall orientation, door centre, margin).
- E3 runs without BatchNorm (with a BN-on reference).
- Rotation head outputs logits (`F.cross_entropy`).
- Latent optimization anchor changed from `||z||^2` to `||z - z0||^2`; acceptance test changed from "monotone" to "final below initial".
- TV unit described as a floor-standing unit (`needs_access: false`).
- Added G0 (procedural generator) as a reference, cost per valid layout, and diversity ratio to G0 (replaces "not lower than B2").
- Held-out sets sampled separately and renamed; out-of-range set added.
- Experiments use one factor at a time with one-seed screening; headline runs happen after a configuration freeze.
- Rejection rate is logged by room area and item count; `f_max` is calibrated.
- Pinned furniture prioritised over the bedroom; bedroom moved to P2. CNN surrogate (M3, E8b) added as a P1 experiment.

**v1.2 (second technical review)**
- Symmetric items (`rot_symmetry > 1`): access checked on every equivalent face, front-face strip drawn on all equivalent faces, orientation excluded from `S_align`. Without this, valid layouts would fail H4 after canonicalization.
- Removed the claim that the CNN is a cheap pre-filter (measured: about break-even with the rule check). The evaluator is justified by the CNN study (E9a) and the surrogate experiment (E8b).
- G0 is excluded from the validity comparison (goal G2 covers B1 and B2 only), its raw valid rate is its acceptance rate per attempt, and the text now says G0 is likely cheaper per valid layout than M2. Added the G0-pin baseline for E12.
- E3 split into E3a (BatchNorm on, decides the deployed activation) and E3b (BatchNorm-off diagnostic). E9 split into E9a (P0) and E9b (P1).
- Corrected the pixel-size claim (0.005 m² is about 1.3 pixels of area, but a strip about a tenth of a pixel wide), the coincident-centre test (assert finite), the door distances (centre 0.65 m, edges 0.2 m), and the nightstand ordering (along the bed's lateral axis).
- Fixed leftovers: training overlap term uses penetration depth, reachability metric divides by items that need access, bedroom marked P2, G0 added to the repository structure and the compare tab.
- Rasters are built on the fly on the GPU (precomputed Set B would need about 16 GB).

**v1.3 (third technical review)**
- Pins: position and optional facing. The facing overwrites the pinned item's rotation before optimization, and the position is snapped exactly before the checks, so the pin is exact by construction. E12 metrics changed accordingly and results are split by which item is pinned.
- Hardware: the evaluator needs the GPU (measured 16 s vs 140 s per epoch); the CPU fallback holds for the CVAE only, with a reduced evaluator setting otherwise.
- Reproducibility: deterministic-mode seed helper (three settings) added; final checks are exact equality on the same machine, with a stated tolerance across machines.
- Leftover wording fixed (see PRD, Architecture and Development Plan change notes).
