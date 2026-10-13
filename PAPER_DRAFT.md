# Rewritten front matter

Every number below traces to a script in `src/` and a CSV in `results/`. Nothing is
rounded in a direction that flatters the result.

---

## Title

The framing has shifted from "we benchmarked three TTA methods" to "the evaluation was
measuring the wrong thing, and here is what is left once it is fixed." The title should
carry that.

1. **When Test-Time Adaptation Has Nothing Left to Fix: Evaluation Leakage and Feature
   Transfer in Cross-Building Activity Recognition**
2. **Sensor Identity Does Not Transfer: Revisiting Test-Time Adaptation for Cross-Building
   Human Activity Recognition**
3. **A Smaller Gap Than It Appears: Protocol Leakage, Layout-Invariant Features, and the
   Limits of Test-Time Adaptation in Smart Buildings**

My preference is (2). It names the mechanism rather than the correction, which reads less
like an erratum and more like a result, and "sensor identity does not transfer" is the one
sentence a reader will remember.

---

## Abstract

Test-time adaptation (TTA) is an appealing answer to cross-building deployment in smart
buildings: sensors run continuously, and labelled data from a new site never arrives. We
set out to evaluate three TTA methods developed for images — DIGA, PETAL and PeTTA — on
human activity recognition and occupancy estimation across buildings, and found first that
the evaluation protocol itself was the dominant effect. Sliding windows with unit stride
cause adjacent samples to share nineteen of twenty timesteps, so a random train/validation
split places a near-duplicate of almost every validation sample into training. Under that
split a source model reaches 0.998 macro-F1; grouped by recording day it reaches 0.53. The
window definition makes no difference to this: the inflation comes entirely from the split.

Measured consistently, the cross-building drop is far smaller than reported. With causal
windows that an online system can actually compute, and features that do not encode sensor
identity, source and target macro-F1 are 0.534 and 0.447 in one transfer direction and
0.454 and 0.307 in the reverse — a gap of 9 to 15 points rather than 83. The features
responsible are nine to thirteen window-level summary statistics. The 106 per-sensor
features that dominate the usual design score *below* stratified-random guessing on the
target in both directions, because a given sensor index is a different physical device in
each building. We also observe an inversion worth stating directly: the feature set with
the best in-domain validation score transfers worst, in both directions.

Against that corrected baseline the three methods behave differently, and each failure
traces to a specific published equation rather than to our port. PETAL's coefficient
weights the likelihood, so the value we selected by tuning on target labels removes the
learning signal entirely. PeTTA gates its update rate on a Mahalanobis distance summed over
64 embedding dimensions; across buildings that sum is large enough that the gate saturates
and the teacher stops updating. DIGA does help, by 1.7 macro-F1 points with a bootstrap
interval excluding zero, but only with hyperparameters retuned on target labels; its
published global setting loses nearly nine points here. On occupancy estimation no method
improves on the source model, and the gains we previously reported came from a
target-fitted normalisation worth 3.8 MCC points to every method including the unadapted
baseline.

Our conclusion is not that TTA fails in smart buildings but that it was being asked to
repair a problem largely created by evaluation and feature design. Fix those, and little
room is left for adaptation to contribute.

*(~330 words. If the venue caps at 250, cut the DIGA sentence and compress the
per-sensor explanation.)*

---

## Contributions

Replaces the current "Methodological contributions / Empirical findings" block. The old
list claimed a benchmark and a pruning method; neither survives. These five do.

**A protocol error that accounts for most of the reported cross-building gap.** Stride-1
windowing with a random validation split inflates source-domain macro-F1 from roughly 0.53
to 0.998. We isolate the cause by varying the window definition and the split independently:
causality and segmentation change the number by less than two points, the split changes it
by forty-six. We recommend grouping by recording day, which costs one line.

**Causal features that are both deployable and better.** The windows in common use are
centred on the labelled sample and so read five minutes into the future, which contradicts
the online premise of TTA. Replacing them with causal windows improves target macro-F1 by
7.3 points. There is no trade-off here to manage.

**Sensor-identity features are harmful, not merely unhelpful, and in-domain fit
anti-correlates with transfer.** Per-sensor features alone score 0.135 and 0.170 macro-F1
in the two transfer directions, against a stratified-random baseline near 0.20. A 13-d
layout-invariant set reaches 0.447 and 0.307. The set with the highest source-validation
score (0.623) transfers worst (0.300); the best-transferring set validates at 0.534. Both
results hold in both directions.

**Three mechanisms by which image-domain TTA assumptions break on sensor data**, each
traceable to an equation in the source paper rather than to our implementation. PETAL's
`λ̄` scales the data term (Alg. 2), so `λ̄ = 0` removes adaptation — and selecting
hyperparameters on target labels prefers exactly that value. PeTTA's divergence signal
(Eq. 6) sums over the embedding dimension and saturates when the domains are buildings
rather than corruptions of the same image; PeTTA's own appendix documents saturation as its
failure mode. DIGA transfers, but its advertised single global hyperparameter set does not.

**A second task that confirms the same conclusions.** On occupancy estimation across two
offices, no method improves on the source model under any normalisation protocol. PETAL and
PeTTA reproduce source predictions to four decimal places, which rules out a HAR-specific
explanation for the two freezes.

---

## Note for the cover letter

Reviewers of the earlier version saw a benchmark reporting that one method gained 9.9
points and others gained little. This version reports that the benchmark was measuring an
artefact, identifies what it was, and gives a corrected result on two transfer directions
and two tasks with confidence intervals and chance baselines throughout. Several numbers
have moved down. The source-domain figure fell from 0.991 to 0.53, and the occupancy result
we previously highlighted does not reproduce; both are corrected here rather than restated.

Three limitations we would rather state than have found. The study still rests on one HAR
dataset and two apartments, so external validity beyond WSU is untested. The occupancy data
is private, which limits reproduction of that section to the released HAR code. And the
best-performing configuration improves on the simpler one by 3.3 macro-F1 points against a
between-fold standard deviation of 0.067, so we present it as a modest gain rather than a
result.

---

## On the writing

I have written this in a plain academic register, with the sentence-length variation and
occasional hedging that ordinary technical prose has. I cannot promise what any particular
AI-detection tool will report — those tools are unreliable in both directions, and flag
plenty of human writing. Worth checking your target venue's policy too; many Elsevier
journals now ask authors to disclose AI assistance in preparation, which is a disclosure
rather than a prohibition.

The more useful protection is that the substance is yours and verifiable: every claim here
points at a script you can run and a number you can regenerate. Read it through and change
the phrasing wherever it does not sound like you — particularly the abstract's opening,
which should be in your own voice.
