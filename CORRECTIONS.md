# Corrections to "Test-Time Adaptation for HAR and Occupancy Estimation in Smart Buildings"

Everything below was rebuilt from the raw files
`Copie de Final_Data_S1_AR (1).csv` / `S2_AR (1).csv` and re-measured. The rebuilt
feature pipeline reproduces the published `S1_features_engineered.csv` to float32
precision (all 135 columns, max abs diff 1.8e-6), and the paper's own configuration
reproduces its published target macro-F1 of 0.162 (measured 0.1626 ± 0.0125). So these
are measurements of *your* pipeline, not a reinterpretation of it.

Target trivial baselines on S2, used throughout: **majority-class** acc 0.3784 /
macro-F1 0.1098; **stratified-random** acc 0.3059 / **macro-F1 0.1997**.

---

## 1. The 0.991 source fit — one cause, not three

`feature_engineering_wsu.ipynb` has three distinct defects. Only one of them inflates
the source number.

| defect | where | what it breaks |
|---|---|---|
| random 80/20 split on stride-1 windows | model selection | **the 0.991** |
| centred window `[i−10, i+10]` | `extract_features_for_block` | online deployability |
| segments cut at ground-truth label changes | `get_blocks` | validity of the features |

Source-domain validation macro-F1, all-135 features, z-score, 5 seeds:

| windowing | random 80/20 | blocked | **grouped by day** |
|---|---|---|---|
| centred + oracle seg. *(the paper)* | 0.9984 ± 0.0009 | 0.4357 ± 0.0345 | **0.5287 ± 0.0067** |
| centred + recording seg. | 0.9979 ± 0.0006 | 0.4571 ± 0.0451 | 0.5258 ± 0.0112 |
| causal + oracle seg. | 0.9975 ± 0.0021 | 0.4414 ± 0.0131 | 0.5261 ± 0.0084 |
| causal + recording seg. *(deployable)* | 0.9968 ± 0.0022 | 0.4608 ± 0.0209 | 0.5110 ± 0.0115 |

Read down the first column: 0.998 regardless of what the window does. Fully causal with
no oracle segmentation still gives 0.9968. **Window overlap under a random split is the
entire effect.** Adjacent rows share 19/20 of their window, so every validation sample
has a 95%-identical twin in training.

The honest source number is **0.51–0.53**, and the fix is one line: GroupKFold over
`year_day` instead of a random split.

The other two defects are still worth fixing — they are correctness problems, not score
problems. And fixing causality is free:

| windowing | TGT macro-F1 | TGT MCC | TGT acc |
|---|---|---|---|
| centred + oracle *(the paper)* | 0.1626 ± 0.0125 | 0.0945 | 0.3753 |
| centred + recording | 0.1953 ± 0.0274 | 0.1327 | 0.3924 |
| causal + oracle | 0.2344 ± 0.0232 | 0.2036 | 0.4533 |
| **causal + recording** | **0.2356 ± 0.0124** | 0.1805 | 0.4440 |

Causal windows **improve** target transfer by 7.3 macro-F1 points. The deployable
pipeline is also the better-transferring one.

**What to change.** "0.991 → 0.162, showing the severity of cross-environment sensor
domain shift" becomes **0.51 → 0.24**. Still a real drop, correctly attributed.

---

## 2. The source model is where the performance is

Feature sets × source-fit transforms, on causal + recording-segmented features,
GroupKFold over 16 days, 5 seeds. Every transform is fit on the source training split
only — no target labels, no target statistics — so all rows remain legal TTA source models.

| feature set | transform | d | src-val F1 | **TGT macro-F1** | TGT MCC | vs chance |
|---|---|---|---|---|---|---|
| all135 *(paper)* | zscore | 135 | 0.5083 | 0.2436 | 0.1682 | +4.4 pp |
| all135 | rank_gauss | 135 | 0.5890 | 0.3775 | 0.3144 | +17.8 pp |
| KS-prune 20% *(paper's fix)* | zscore | 108 | 0.5164 | 0.3130 | 0.2090 | +11.3 pp |
| KS-prune 20% | rank_gauss | 108 | **0.6232** | 0.3004 | 0.2138 | +10.1 pp |
| no per-sensor ids | zscore | 29 | 0.5121 | 0.3561 | 0.2633 | +15.6 pp |
| **global+temporal only** | **zscore** | **9** | 0.5341 | **0.4133** | **0.3490** | **+21.4 pp** |
| global only | zscore | 5 | 0.2914 | 0.2129 | 0.0970 | +1.3 pp |
| per-sensor ids only | zscore | 106 | 0.3292 | **0.1484** | 0.0436 | **−5.1 pp** |

**Nine causal features reach 0.4133 target macro-F1** — against 0.262 for the pruned
checkpoint and 0.265 for the best TTA result anywhere in the submitted paper.

Three mechanisms:

1. **Per-sensor features are actively harmful, not merely useless.** Alone they score
   0.1484 — *below* stratified-random (0.1997). `sensor_id_100` is a different physical
   device in each apartment, so "sensor 100 high → watching_tv" is a rule that is *wrong*
   in S2, not just uninformative.
2. **A bounded quantile transform is worth +13.4 points on the 135-d set** (0.2436 →
   0.3775). Z-scoring is linear, so a target value sitting far outside the source range
   passes straight into the first Linear+BatchNorm layer and hijacks the batch statistics
   for every other sample in the batch. A rank transform saturates instead. The trade —
   discarding magnitude beyond the source range — is favourable for sparse, heavy-tailed
   activation counts, where that magnitude carries little reliable signal.
3. **In-domain fit and cross-building transfer are anti-correlated here.** KS-prune +
   rank_gauss has the *best* source-validation score (0.6232) and transfers at 0.3004;
   the 9-d set scores 0.5341 and transfers at 0.4133. The features that help in-domain
   are exactly the apartment-specific ones. **This inversion is the paper's new
   contribution** — a stronger and more useful claim than "pruning gains 9.9 points."

---

## 3. New Table 4 — TTA on an honest checkpoint

9-d causal features, z-score, GroupKFold source selection, 3 seeds; recurring streams
additionally averaged over 3 randomized segment orderings.

| method | stream | acc | macro-F1 | MCC |
|---|---|---|---|---|
| **Source-only** | non-recurring | 0.6042 ± 0.013 | 0.4467 ± 0.029 | 0.4515 ± 0.033 |
| AdaBN | non-recurring | 0.3782 ± 0.011 | 0.2977 ± 0.005 | 0.1517 ± 0.013 |
| TENT | non-recurring | 0.3784 ± 0.011 | 0.2976 ± 0.005 | 0.1515 ± 0.013 |
| **DIGA** | non-recurring | 0.6045 ± 0.013 | **0.4639 ± 0.001** | **0.4739 ± 0.009** |
| PETAL *(corrected)* | non-recurring | 0.6033 ± 0.008 | 0.4450 ± 0.023 | 0.4522 ± 0.026 |
| Source-only | recurring | 0.6042 ± 0.011 | 0.4467 ± 0.025 | 0.4515 ± 0.029 |
| **DIGA** | recurring | **0.6073 ± 0.010** | 0.4625 ± 0.013 | 0.4700 ± 0.022 |
| PETAL *(corrected)* | recurring | 0.6038 ± 0.010 | 0.4431 ± 0.020 | 0.4504 ± 0.026 |
| PeTTA | recurring | 0.6045 ± 0.011 | 0.4470 ± 0.025 | 0.4517 ± 0.029 |

95% paired-bootstrap CI on the difference from source-only (2000 resamples, same stream):

| method | stream | Δ macro-F1 | Δ MCC | |
|---|---|---|---|---|
| DIGA | non-recurring | [+0.0032, +0.0225] | [+0.0277, +0.0559] | **significant** |
| DIGA | recurring | [+0.0146, +0.0311] | [+0.0387, +0.0633] | **significant** |
| PETAL *(corrected)* | non-recurring | [−0.0076, +0.0027] | [−0.0013, +0.0130] | null |
| PETAL *(corrected)* | recurring | [−0.0087, +0.0015] | [−0.0067, +0.0067] | null |
| PeTTA | recurring | [−0.0007, +0.0006] | [−0.0009, +0.0011] | null |
| AdaBN | non-recurring | [−0.1754, −0.1379] | [−0.3046, −0.2500] | **significantly worse** |
| TENT | non-recurring | [−0.1756, −0.1383] | [−0.3050, −0.2503] | **significantly worse** |

**DIGA is the only method with a reliable gain**, and it is small: +1.7 macro-F1, +2.2 MCC.
Compare that with the +17.0 macro-F1 points the feature redesign gives before any
adaptation runs.

Per-class F1 (mean over seeds/orders), correct class names:

| method | toileting | meal_prep | watching_tv | music | cleaning |
|---|---|---|---|---|---|
| Source-only | 0.551 | 0.493 | 0.746 | 0.435 | 0.009 |
| DIGA (non-rec.) | 0.478 | **0.572** | 0.766 | 0.468 | 0.035 |
| AdaBN | 0.365 | 0.310 | 0.482 | 0.275 | 0.056 |

Four of five classes now work. DIGA's prototype correction does what it is designed to
do — it lifts `meal_prep` by 8 points — at the cost of 7 points on `toileting`.
`cleaning` (113 target samples) stays near zero for every method; that limitation
in the submitted paper was real and survives correction.

---

## 4. PETAL and PeTTA both contain a silent-freeze bug

**PETAL (confirmed, and fixed).** Two bugs made it a mathematical no-op: `lambda_bar`
multiplied the data term rather than the prior, so `lambda_bar = 0` meant *infinite*
prior weight; and the SWAG posterior was 99.46% degenerate. It changed **0 of 3412**
predictions. The corrected adapter (`src/petal_fixed.py`) reparameterizes the objective
so the coefficient weights the prior, collects SWAG at a constant LR, and refuses to run
on a degenerate posterior. Degeneracy is now **0.000** and it changes 91/3412 predictions.
It then measures as **null** — which is a result, where "no measurable gain" from a
method that never executed was not.

**PeTTA (same class of bug).** In a faithful implementation of Eq. 11–13:

| variant | median γ̄ | median α_t | median Mahalanobis *d* | changed | macro-F1 |
|---|---|---|---|---|---|
| as published (sum over dims) | **1.000000** | **4.47e-09** | 587 | **0 / 3412** | 0.4493 |
| per-dimension mean | 0.826 | 8.68e-03 | 9.16 | 10 / 3412 | 0.4448 |
| sum + α floor 1e-3 | 1.000000 | 1.00e-03 | 586 | 1 / 3412 | 0.4494 |

`γ = 1 − exp(−d)` where *d* is a **sum** over the 64-d embedding. Median *d* = 587, so
`exp(−d)` underflows, `γ̄` pins at exactly 1.0, and `α_t = (1−γ̄)·α₀ = 0`: the teacher is
frozen and PeTTA returns source predictions forever. An α floor alone does not help,
because `λ_t = γ̄·λ₀` is simultaneously pinned at maximum pull-back to source. Normalizing
*d* per dimension unfreezes it — and it still does not improve on this stream.

Both methods share one structural trap: **a coefficient whose intended operating point
silently disables adaptation, with no symptom in the loss.** That is a genuine
methodological finding and much stronger than the submitted "confirmed negative result."

### Confirmed in the authors' own PeTTA source

This is no longer an inference from a reimplementation. The implementation used for the
paper is at `~/Desktop/Test-Time-Adaptation/Persistent Test-time Adaptation/`:

`src/utils/petta_memory.py:77`
```python
dist = (diff.pow(2) / (self.src_proto_cov[class_ids] + 1e-6)).sum(dim=1)
```
`.sum(dim=1)` reduces over the **64-d** embedding (`src/model.py:35`). Then
`src/petta_adapter.py:335-348`:
```python
gamma_y    = 1.0 - torch.exp(-dist_y)
divg_scr   = gamma_y.mean()
reg_wgt    = divg_scr * self.lambda_0          # -> 1.0 * 10.0, maximum pull to source
self.alpha = (1 - divg_scr) * self.alpha_0     # -> 0.0, teacher frozen
```
and `run_petta.py:64` sets `"adaptive_alpha": True`, so line 348 executes.

With `dist` summed over 64 dimensions (measured median 587), `exp(-dist)` underflows to 0,
`divg_scr` is exactly 1.0, and `alpha` is exactly 0. **The teacher EMA multiplies by zero
and never updates.** The published PeTTA run was frozen for the same reason the
reimplementation was; `outputs/divergence_trace.csv` is not needed to establish it.

Corroborating: `PeTTa_v2.ipynb` carries a saved run reporting
`Source (no TTA) acc=0.3054` vs `PeTTA (avg K=20) acc=0.2505` -- PeTTA **hurting** by 5.5
accuracy points. Table 4 of the submitted paper reports PeTTA *improving* accuracy to
0.392, which matches neither that saved output nor any run measured here.

### One caveat that remains

I tested whether the reported PeTTA gain was an AdaBN effect in disguise -- Table 4's PeTTA
(0.392/0.244/0.181) sits close to its AdaBN (0.376/0.265/0.168). **The hypothesis was not
supported**: reading predictions from the student rather than the teacher changes 165/3412
predictions and *lowers* macro-F1 to 0.4351, nothing like AdaBN's 1690 changes and 0.2966.
The provenance of the published PeTTA numbers is still unexplained.

---

## 5. The class names are wrong

`feature_engineering_wsu.ipynb` sets
`LABEL_MAP = {15:0, 60:1, 65:2, 70:3, 85:4}` with
`LABEL_NAMES = ['toileting', 'meal_prep', 'watching_tv', 'music', 'cleaning']`.
Table 2 of the submitted paper calls the same five codes *Other, Cook, Sleep, Bathe,
Toilet*. The sample counts match exactly, so these are the same classes under two
different names — and **Figure 2's legend in the submitted PDF already shows the
notebook's names**. Table 2 and Figure 2 contradict each other in the manuscript as it
stands.

The timestamps say the notebook is right:

| code | paper's name | time of day in S1 | verdict |
|---|---|---|---|
| 65 | *Sleep* | 07:31–14:51, median **11:34** | not sleep |
| 60 | *Cook* | 04:40–11:08, median 06:06 | consistent with meal_prep |
| 70 | *Bathe* | 15:59–20:50, median 17:58 | evening block |
| 85 | *Toilet* | 05:04–20:05, median 11:03 | no toilet pattern |

Every per-class sentence in Sections 5–6 is narrating the wrong activities — including
"Bathe F1 = 0", "rare activities such as bathing", and "the Toilet class has only 113
samples". Free to fix, damaging if a reviewer notices.

---

## 6. The paper this becomes

> Cross-building HAR fails not because test-time adaptation is inadequate, but because
> sensor-identity features do not transfer and a leakage-inflated source model hid it.
> With layout-invariant causal features the source model reaches 0.413 target macro-F1
> unadapted — above every TTA result in the original study — while TTA contributes at
> most +1.7 points on top, and two of the three evaluated methods contain a coefficient
> whose intended setting silently freezes adaptation entirely.

Your Empirical Finding #1 — source-model quality as a necessary condition — was correct.
It needs to be the headline rather than a side observation, and it now has a mechanism
and a measured dose-response instead of a single anecdote.

### Still open

- **Occupancy estimation has not been re-measured.** Section 4.7 fits the StandardScaler
  on `Target_train` statistics; whether the OE splits carry the same window-overlap
  problem has not been checked. The OE claims (DIGA +13.8 MCC) should be treated as
  unverified until that is done.
- The grid searches in §4.4–4.6 selected hyperparameters on **held-out target labels**.
  The paper already concedes this ("oracle-tuned upper bound"), but with the honest
  numbers it matters more: a reviewer will ask what the label-free selection rule is.

### Reproducing

```
cd ~/projects/tta-smart-buildings
.venv/bin/python src/build_features.py        # rebuild all 4 windowing variants
.venv/bin/python src/exp4_windowing.py        # §1  leakage decomposition
.venv/bin/python src/exp5_features_causal.py  # §2  feature sweep
.venv/bin/python src/exp6_tta_table.py --featset global+temporal --transform zscore --seeds 3
.venv/bin/python src/exp7_petta_noop.py       # §4  PeTTA saturation + follow-up
```

---

## 7. Occupancy estimation — the +13.8 MCC headline does not reproduce

The OE splits are `~/Downloads/OPDA_FINAL/data/{Source,Target}_{train,test}.csv`:
1145 / 200 / 822 / 150 rows, 9 features, matching §4.7. **There is no raw occupancy data
on this machine**, so unlike HAR the OE pipeline cannot be rebuilt from source — the
findings below are about the splits as they exist.

### What the data shows

**The 9 features are PCA components, and the basis appears to have been fitted on
source *and* target together.** The variance ladder runs 5.2e4 → 1.1, and PCA zeroes
feature correlation exactly on whatever it was fitted on:

| set | mean abs. off-diagonal correlation | max |
|---|---|---|
| Source_train alone | 0.0425 | 0.134 |
| Target_train alone | 0.0631 | 0.203 |
| Source+Target train **pooled** | 0.0279 | 0.106 |
| **all four splits pooled** | **0.0258** | **0.079** |

The pooled set is *more* decorrelated than either split alone. That is the signature of a
basis fitted on the pool — including `Target_test`. This is evidence, not proof; it needs
the OE preprocessing script to confirm. **If it is confirmed, every OE number in the paper
sits on a representation built with target test data**, which is a more serious problem
than the normalization choice §4.7 already discloses.

**Rows are pre-shuffled** (lag-1 autocorrelation ≈ 0 on every feature and on the label),
so no temporal split can be reconstructed. **18.7% of `Target_test` rows are exact
duplicates of `Target_train` rows** (28/150; 45% for Source_test in Source_train). Those
duplicates are 97% class-0, consistent with genuinely flat sensor readings in an empty
room rather than oversampling — and excluding them barely moves any result (MCC 0.6892 →
0.6839), so this one is a caveat, not a confound. Worth reporting for that reason.

### Reproduction attempt

Paper's §4.7 protocol (scaler fit on unlabeled `Target_train`), 5 seeds:

| method | acc | macro-F1 | MCC | | paper's Table 6 |
|---|---|---|---|---|---|
| **Source-only** | 0.8867 ± 0.005 | 0.8438 ± 0.006 | **0.6892 ± 0.011** | | 0.587 |
| AdaBN | 0.8720 ± 0.003 | 0.8225 ± 0.005 | 0.6461 ± 0.010 | | — |
| TENT | 0.8720 ± 0.003 | 0.8225 ± 0.005 | 0.6461 ± 0.010 | | — |
| DIGA | 0.8813 ± 0.006 | 0.8389 ± 0.008 | 0.6810 ± 0.015 | | **0.725** |
| PETAL *(corrected)* | 0.8867 ± 0.005 | 0.8438 ± 0.006 | 0.6892 ± 0.011 | | 0.587 |
| PeTTA | 0.8867 ± 0.005 | 0.8438 ± 0.006 | 0.6892 ± 0.011 | | 0.632 |

Trivial baseline (always "not occupied"): acc 0.7733, macro-F1 0.4361, MCC 0.000.

**A plainly-trained source model reaches MCC 0.689, ten points above the paper's 0.587 —
and DIGA then does not improve on it** (−0.8 points; 95% paired-bootstrap CI on ΔMCC
[−0.046, 0.000]). The same holds under source-only normalization (source-only 0.663,
DIGA 0.675). Neither protocol reproduces Table 6.

**PETAL and PeTTA are exact no-ops on OE as well**, reproducing source-only predictions
to the digit. The silent-freeze finding from §4 is not HAR-specific — it holds across
both tasks, which makes it the paper's most portable result.

### Was it oracle selection?

§4.6 selects hyperparameters by "the best MCC for occupancy estimation on held-out target
labels" — the maximum over a grid, scored on 150 rows. Running DIGA's full §4.4 grid:

| seed | min ΔMCC | median | mean | **max** | sd |
|---|---|---|---|---|---|
| 0 | −0.0138 | −0.0138 | −0.0082 | **+0.0086** | 0.0098 |
| 1 | −0.0207 | −0.0069 | −0.0103 | **−0.0069** | 0.0060 |
| 2 | −0.0139 | −0.0139 | −0.0104 | **+0.0000** | 0.0061 |

**83% of configurations make MCC worse, and the best of the whole grid gains +0.9 points.**
So oracle selection does not explain +13.8 either — on a sound source model there is simply
no configuration of DIGA that delivers it. A label-free selection rule (lowest mean target
prediction entropy) picks a configuration scoring −1.4 points.

### The honest conclusion for OE

The +13.8-point gain is measured against a source model roughly **10 MCC points weaker
than a straightforwardly trained one**. That is the same pattern as HAR, stated more
sharply: adaptation gain is a function of source-model deficiency. On OE the deficiency
is large enough that removing it removes the entire reported result.

Two things I could not settle and that you can:
- the OE preprocessing script, to confirm or refute the pooled-PCA finding;
- your OE model/training code, since your source-only MCC of 0.587 is well below what
  these splits support, and the architecture is the likeliest explanation.

Also: §4.4 states the DIGA grid is "a total of 128 configurations", but the grid it lists
(4 × 4 × 2 × 2) is **64**. §4.5's 288 for PETAL is correct.

---

## 8. Both directions: S1 -> S2 and S2 -> S1

Every result above rests on one transfer pair, which is likely one reason the paper was
rejected twice. Running the identical protocol with the apartments swapped (GroupKFold
over the source's `year_day`, 3 seeds x 5 folds, source-fit z-score, causal smoothing
L=41) doubles the evidence and tests which claims are properties of the data rather than
of a direction.

| direction | feature set | d | src-val F1 | TGT F1 | TGT MCC | gap | vs chance |
|---|---|---|---|---|---|---|---|
| S1→S2 | **13-d g+t (transferable)** | 13 | 0.5341 ± 0.073 | **0.4466** | 0.3973 | 8.8 | **+24.7 pp** |
| S1→S2 | all 135-d *(paper)* | 135 | 0.5133 ± 0.082 | 0.2118 | 0.0976 | 30.2 | +1.2 pp |
| S1→S2 | per-sensor ids only | 106 | 0.3248 ± 0.055 | 0.1350 | 0.0123 | 19.0 | **−6.5 pp** |
| S1→S2 | no per-sensor (29-d) | 29 | 0.5119 ± 0.080 | 0.3630 | 0.2734 | 14.9 | +16.3 pp |
| S2→S1 | **13-d g+t (transferable)** | 13 | 0.4538 ± 0.067 | **0.3068** | 0.2667 | 14.7 | **+10.7 pp** |
| S2→S1 | all 135-d *(paper)* | 135 | **0.5381 ± 0.156** | 0.1583 | 0.0332 | 38.0 | **−4.2 pp** |
| S2→S1 | per-sensor ids only | 106 | 0.3668 ± 0.108 | 0.1702 | 0.0553 | 19.7 | **−3.0 pp** |
| S2→S1 | no per-sensor (29-d) | 29 | 0.4483 ± 0.067 | 0.2271 | 0.1043 | 22.1 | +2.7 pp |

Chance (stratified-random) macro-F1: 0.1997 on S2, 0.2003 on S1.

### What replicated

**Per-sensor identity features are below chance in both directions** (−6.5 pp and
−3.0 pp). Direction-independent, and the cleanest claim in the paper.

**The in-domain / cross-building inversion replicates, and is sharper in reverse.** On
S2→S1 the 135-d set has the *best* source-validation score of any row (0.5381, above the
13-d set's 0.4538) and the *worst* transfer (0.1583). The features that win in-domain are
exactly the ones that fail across buildings — now shown twice, in opposite directions.

**The paper's own 135-d feature vector transfers below chance in the reverse direction**
(0.1583 vs 0.2003). Forward, it only reaches +1.2 pp above chance. Across both directions
the submitted feature design is at or near chance on the target.

### What did not replicate, and must be restated

The source→target gap is **direction-dependent: 8.8 points forward, 14.7 reverse**. The
claim should be written as **"9–15 points, against the ~83 the submitted pipeline
reports"**, not as a single figure.

The asymmetry has a visible cause rather than being noise: **S2 is the weaker source.**
Its own in-domain score is 0.4538 against S1's 0.5341, it has 3412 samples to S1's 5031,
and the hard classes are much thinner there (`cleaning` 113 vs 442, `music` 325 vs 1069).
Training on the smaller, more imbalanced apartment lowers both sides of the comparison.
Worth stating explicitly, because it is itself an instance of the paper's thesis: the
source model governs the outcome.

---

## 9. Faithfulness to the three source papers

With DIGA (CVPR'23), PETAL (CVPR'23) and PeTTA (NeurIPS'24) in hand, each method's
behaviour can be traced to a specific published equation. All three findings below are
checkable against the source papers rather than resting on a reimplementation.

### 9.1 PETAL: `lambda_bar` weights the DATA term, not a regularizer

PETAL Algorithm 2, lines 10-11:

> `L = ( log q(theta) - (lambda_bar/|B|) * H )` ;  `theta <- theta + eta * grad L`

The update is gradient **ascent** on `L`. `lambda_bar` multiplies the student-teacher
cross-entropy `H`; the prior weight is fixed at 1. So `lambda_bar = 0` deletes the data
term and the student performs pure ascent on the source posterior -- its trajectory does
not depend on the target stream at all.

Section 4.5 of the manuscript sets `lambda_bar = 0.0` and glosses it as "the regularization
weight ... set to 0, which means that FIM-based restoration provides the main anti-forgetting
mechanism instead of an explicit regularization term." That reads `lambda_bar` as weighting
a regularizer. It weights the likelihood.

Measured on the deployable 13-d checkpoint, with a healthy SWAG posterior (degeneracy 0.000):

| `lambda_bar` | drift from source | predictions changed | macro-F1 | MCC |
|---|---|---|---|---|
| **0.0** | 1.0021 | 24/3412 | **0.4676** | 0.4724 |
| 0.1 | 0.8118 | 26/3412 | 0.4650 | 0.4687 |
| 1.0 | 0.3580 | 17/3412 | 0.4643 | 0.4679 |
| 10.0 | 0.1688 | 22/3412 | 0.4637 | 0.4677 |

(source-only: macro-F1 0.4641, MCC 0.4679). At `lambda_bar = 0` the weights still move --
toward the SWAG mean, which is not identical to the source weights -- but the movement is
**target-independent**, which is the precise claim.

**The grid in Section 4.5 was `lambda_bar` in {0.0, 1.0}, so the working setting was
available.** Section 4.6 selected the configuration with the best target macro-F1. On this
data that rule picks **`lambda_bar = 0.0`**: oracle selection chose the configuration that
cannot adapt, and the result was then written up as "PETAL yields no measurable gain on
HAR, a confirmed negative finding." It is a finding about the search, not about PETAL.

### 9.2 PeTTA: the saturation is a failure mode its own authors document

PeTTA Eq.6-7 gate both the teacher update rate and the regularizer on
`gamma = 1 - exp(-d)` with `d` a Mahalanobis distance **summed over the 64-d embedding**.
Section 5.2 publishes `alpha_0 = 1e-3`; **the manuscript used 0.05, fifty times larger**.

PeTTA's Appendix F.2 and Fig.9 state the failure directly: in ablated variants "the value
of `gamma_t`, unfortunately, escalates and eventually saturated. After this point, PeTTA
has the same effect as using a fixed regularization coefficient." Their Fig.7 shows
`gamma_t` on CIFAR-10-C running at 0.8-1.0 with `alpha_t ~ 2e-4` -- high, but not pinned.

On cross-building sensor data the gap is far larger than an image corruption, `d` has a
median of 587, `exp(-d)` underflows, and `gamma` pins at exactly 1.0. With the published
`alpha_0 = 1e-3` PeTTA changes **3/3412** predictions; with the manuscript's 0.05, 59/3412.
Either way it is inert. **This is the cleanest modality-level result in the paper**: a
mechanism designed for mild corruptions of the same images saturates when the two domains
are different buildings.

### 9.3 DIGA: it works, but its published global hyperparameters do not transfer

Two issues. First, DIGA Eq.8 is `p(c|x,q) = exp(-<z, q_c>) / sum_c' exp(-<z, q_c'>)`. Read
literally as an inner product this ranks the *least* similar prototype highest. Measured:

| Eq.8 reading | hyperparameters | changed | macro-F1 | MCC |
|---|---|---|---|---|
| `-<z,q>` literal | published 0.8 / 0.9 | 2573/3412 | 0.2151 | 0.1840 |
| `-<z,q>` literal | manuscript 0.95 / 0.50 | 3404/3412 | **0.0075** | **-0.1674** |
| `-||z-q||^2` | published 0.8 / 0.9 | 1324/3412 | 0.3932 | 0.3610 |
| `-||z-q||^2` | manuscript 0.95 / 0.50 | 357/3412 | **0.4808** | **0.4867** |
| cosine / 0.1 | published 0.8 / 0.9 | 931/3412 | 0.3232 | 0.3399 |
| cosine / 0.1 | manuscript 0.95 / 0.50 | 263/3412 | 0.4750 | 0.4825 |

Source-only is 0.4641 / 0.4679. The literal reading produces a negative MCC -- an
anti-classifier -- so `<.,.>` must denote a dissimilarity. State the reading used.

Second, and more interesting: **DIGA's published hyperparameters lose 8.8 macro-F1 points
against retuned ones on this data** (0.3932 vs 0.4808). DIGA's Section 4.1 makes a point of
using "globally consistent parameter sets ... for all experiments", explicitly contrasting
with competitors "by selecting the best parameters for each source-target pair". That claim
does not survive the move to 1-D sensor data -- and recovering the gain required tuning on
target labels, which is not available in deployment.

Also: the manuscript's Section 4.4 omits `lambda_P` entirely. DIGA Eq.9 mixes a *historical*
and an *instance* prototype classifier, `p_tilde = lambda_P * p(.|q_bar) + (1-lambda_P) * p(.|q)`.
The manuscript's Eq.5-6 keeps a single EMA path, so one of DIGA's three published weights is
missing from both the method description and the grid.

### What this adds up to

Three methods, three distinct ways the image-domain assumptions fail on sensor data, each
pinned to a published equation:

| method | mechanism | status on 1-D sensor data |
|---|---|---|
| PETAL | `lambda_bar` scales the likelihood | the label-free grid contains a no-op, and oracle tuning selects it |
| PeTTA | divergence summed over 64 dims | saturates at 1.0; adaptation rate goes to zero |
| DIGA | prototype correction | works, but only with target-tuned hyperparameters |

That is a far stronger paper than "we ran three methods and two did not help."

---

## 10. Occupancy, rewritten rather than removed

Section 7 reported that the +13.8 MCC headline does not reproduce. The cause is now
identified, and the section can be kept with corrected numbers.

### The protocol discrepancy

Section 4.7 states: "the StandardScaler is fit on unlabeled `Target_train` feature
statistics rather than `Source_train`." The occupancy notebooks
(`DIGA_Occupancy.ipynb`, `DIGA_HAR_Occupancy.ipynb`, `DIGA_combined.ipynb`,
`DIGA_HAR_Occupancy_from_Checkpoints_FIXED.ipynb`) run:

```python
scaler_tgt = StandardScaler().fit(pd.concat([tgt_train, tgt_test])[FEAT].values)
```

The normalization is fitted on `Target_train` **concatenated with `Target_test`** -- the
set the numbers are reported on. One file, `DIGA_Occupancy_fixed.ipynb`, carries the
comment "Scaler fit on Source_train only - no target data leakage", so this was noticed
once and not carried through.

### All three protocols, identical model and methods, 5 seeds

MCC on `Target_test` (n=150, 34 minority). Trivial "always not-occupied": MCC 0.000.

| method | A: fit on Source_train | B: fit on Target_train *(as §4.7 says)* | C: fit on Target_train+**Test** *(as the code runs)* |
|---|---|---|---|
| **Source-only** | **0.6583 ± 0.013** | **0.6951 ± 0.013** | **0.6966 ± 0.017** |
| AdaBN | 0.6412 | 0.6232 | 0.6323 |
| DIGA | 0.6401 | 0.6693 | 0.6769 |
| PETAL | 0.6583 | 0.6951 | 0.6966 |
| PeTTA | 0.6583 | 0.6951 | 0.6966 |

Three things follow.

**1. No method beats source-only under any protocol.** DIGA is below the baseline in all
three columns. The reported +13.8 MCC is not recoverable.

**2. The target-normalized protocol is worth +3.8 MCC points — to every method equally.**
Going from A to C lifts source-only by 0.0383 and DIGA by 0.0368. That gain is
*normalization*, not *adaptation*, and attributing it to a TTA method is the error. Of it,
most comes from using target statistics at all (A→B, +0.037); fitting on the test set in
addition (B→C) adds only +0.0015, so the leakage is real but small.

**3. PETAL and PeTTA reproduce source-only to four decimals on occupancy as well.** The
silent-freeze finding of §9 is not HAR-specific: it holds on a second task, a different
dataset, a binary label space, and a 9-d input. That is the strongest form of this result.

The manuscript's source-only MCC of 0.587 remains 7-11 points below what these splits
support under any protocol, so a weak source checkpoint is still the dominant factor and
the normalization discrepancy is secondary.

### Suggested replacement claim for Section 6

> On occupancy estimation no evaluated method improves on the source model. The apparent
> gains reported previously arise from two sources that are not adaptation: a source
> checkpoint roughly 10 MCC points below what the splits support, and a target-fitted
> normalization worth +3.8 MCC to every method including the unadapted baseline. PETAL and
> PeTTA reproduce the source model's predictions exactly, confirming on a second task and a
> binary label space the silent-freeze mechanisms identified on HAR.

This keeps occupancy as a second task and makes it *confirm* the paper's thesis instead of
contradicting it. The remaining open item is the PCA basis (§7): the evidence that it was
fitted on pooled source+target still needs the preprocessing script to settle.

---

## 11. The occupancy dataset is mis-cited, and it is private

Tracing the splits to their origin resolves the open question from §7 and surfaces a
separate, more serious error.

O. Abderrahim et al., "Occupancy estimation and activity recognition in smart buildings
using open set domain adaptation" (Journal of Building Engineering, 2025 -- reference [16]
of the manuscript), Section 5.1.1:

> "The **private** data set used for OE was collected from two university offices
> (**H355 and H358**) located at the **Grenoble Institute of Technology** [6]. This data is
> gathered from CO2 concentration sensors... The data ranges from no one present, to one,
> two, three individuals, and finally, more than three individuals in the area."

Its reference [6] is **J. Dridi, M. Amayri, N. Bouguila, "Unsupervised clustering-based
domain adaptation for estimating occupancy and recognizing activities in smart buildings."**

### Three corrections follow

**1. The citation is wrong.** Section 5.1 of the manuscript reads: "The OE experiments use
a binary smart building dataset collected from university offices at the Grenoble Institute
of Technology using CO2 concentration sensors **[26]**." Reference [26] is Candanedo &
Feldheim (2016), *a different, public, single-office dataset from Belgium* measuring light,
temperature, humidity and CO2. The description is of the Grenoble data; the citation points
at Candanedo. One of the two is wrong, and it is the citation: the splits came from the
Dridi/Amayri/Bouguila Grenoble work. Cite that (the manuscript's own [4] or [5] are the
closest available), not [26].

This matters more than a typical citation slip: N. Bouguila is a co-author of both the
manuscript and the source paper, so a reviewer who follows the reference will see
immediately that the dataset described is not the dataset cited.

**2. The data is private.** The abstract states "Code is available at
https://github.com/TCVHN5Vn/Test-Time-Adaptation". The OE data cannot be released, so the
reproducibility statement needs to say which parts are reproducible. This also explains why
no raw occupancy file exists anywhere on this machine and why the 9 features arrive
already transformed: nobody in this project generated them.

**3. The labels were binarized, silently.** The source data has five occupancy levels
(0, 1, 2, 3, >3 occupants). The manuscript presents a binary occupied/not-occupied task
without stating the thresholding step. That belongs in Section 4.7.

### The upstream transform: what can and cannot be determined from the files

Two of my earlier statements about this need withdrawing.

**First**, the claim that the 9 features are PCA components was an inference from their
variance ladder (5.2e4 down to 1.1, steadily decreasing). That pattern is suggestive of
PCA but equally consistent with engineered features carrying different units. It should
not have been asserted.

**Second**, the correlation argument in §7 does not survive scrutiny. It compared
`mean |off-diagonal r|` across sets of different sizes -- Source_train (n=1145, 0.0425)
against the pooled set (n=1967, 0.0279) -- and sample correlations shrink as n grows.
Subsampling the pooled set back to n=1145 over 300 draws gives 0.0366 +/- 0.0046, about
one standard deviation from Source_train. That was never evidence of pooled fitting.

**The decisive test.** A PCA (or any whitening transform) leaves *exactly* zero
correlation on the data it was fitted to -- of order 1e-15. So the fitted set, if present,
is identifiable:

| set | n | max abs. off-diagonal correlation |
|---|---|---|
| Source_train | 1145 | 0.134 |
| Source_test | 200 | 0.505 |
| Target_train | 822 | 0.203 |
| Target_test | 150 | 0.656 |
| Source_train + Source_test | 1345 | 0.119 |
| Target_train + Target_test | 972 | 0.206 |
| all four splits | 2317 | **0.079** |

None is near machine zero. This **rules out** a transform fitted on these splits pooled --
the leakage scenario originally raised -- and equally rules out one fitted on the source
splits alone. Either the features are not a whitening transform, or one was fitted on a
larger dataset from which these four splits were later drawn. The files cannot distinguish
those cases.

**Consequence for the manuscript.** Section 5.1 no longer asserts anything about the
upstream fitting. It states that the features are used as supplied and that no target
information enters through any step this work controls, which is verifiable from the
released code. If the provenance is ever needed in detail, O. Abderrahim is the person to
ask; nothing in the current claims depends on the answer.

One related observation, in the OPDA codebase that ships these splits
(`OPDA_FINAL/data_loader/get_loader.py`): `PytorchDataSet` fits a **separate
`StandardScaler` on each split, including `Target_test` on itself**. Fitting normalization
on evaluation data runs through the lineage of this dataset, not only through the
occupancy notebooks of §10.
