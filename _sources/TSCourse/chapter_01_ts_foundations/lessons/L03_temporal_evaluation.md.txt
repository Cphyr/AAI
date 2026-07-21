```
Author: Cfir Hadar

Tags: Done
```
# Lesson 03 - Evaluation for Temporal Data

```{note}
This lesson is a **lens for the entire course**. Every challenge from here on will ask you which
protocol you used and why. Come back to it.
```

## Motivation

In i.i.d. supervised learning, evaluation is a solved ritual: shuffle, k-fold, report. In temporal
data that ritual produces numbers that are not merely optimistic — they are *meaningless*, because
they measure interpolation of a series you already know, not prediction of one you do not. Almost
every dramatic result that fails to reproduce failed here.

## Protocols

![Evaluation protocols](../../../_static/ts/evaluation_protocols.png)

**Walk-forward / rolling origin.** Repeatedly train on $[1,t]$, test on $(t,t+h]$, advance $t$.
Two variants: *expanding* window (train set grows — right when the process is stable) and *sliding*
window (fixed length — right when it drifts; also gives comparable training sizes across folds).
This is the default for forecasting. Report the distribution of fold errors, not just the mean:
a method that wins on average while losing catastrophically in two folds is usually the wrong
choice.

**Blocked / grouped CV.** For classification over windows or tracks: split into contiguous blocks
and hold out whole blocks, with an **embargo** (a gap of at least the window length plus the label
horizon) on each side of the test block, so that overlapping windows do not straddle the boundary.

**Group-by-entity split.** For track data, the group is the *aircraft / vessel / vehicle*, not the
window and not the track. The same platform flying the same route twice is nearly the same sample.

**Nested selection.** Hyperparameters must be chosen inside the training portion of each fold
(an inner rolling-origin split). Choosing them on the test folds and then reporting those folds is
the most common way a solid protocol still ends up overfit.

## Why random k-fold is wrong

Three separate defects, worth keeping distinct:

1. **Temporal leakage.** Training on $t+1$ to predict $t$ is not a task anyone will ever face.
2. **Autocorrelation.** Neighbouring points are nearly duplicates, so a held-out point has a
   near-copy in training: the estimate measures memorisation, not generalisation.
3. **Distribution mismatch.** Real deployment always tests on a *later* period, whose distribution
   has drifted. Random folds average that away and hide the drift you were supposed to measure.

## Leakage patterns specific to tracks

| Leak | Mechanism | Fix |
| --- | --- | --- |
| **Identity leakage** | same aircraft/vessel (or the same flight, resampled) in train and test | group split by platform ID; deduplicate near-identical tracks |
| **Future context** | features from a centred window, a smoother (RTS), or a resampling filter that uses future samples | causal features only; recompute preprocessing inside each fold |
| **Global normalisation** | mean/std, PCA, or a scaler fitted on all data | fit on train fold only |
| **Label leakage via augmentation** | augmenting before splitting, so an augmented copy of a test track is in training | split first, augment inside training only |
| **Target-derived features** | "time since last event", peak values, or any feature computed with knowledge of the labelled interval | derive features strictly from the past |
| **Imputation across the split** | interpolating gaps over the boundary | impute per fold, causally |
| **Selection leakage** | dropping "bad" tracks by a criterion computed on the whole dataset | define exclusions from training data only |

The diagnostic that catches most of these: **shuffle the labels and retrain**. Any skill above
chance means information is flowing where it should not.

## Comparing two models honestly: Diebold-Mariano

You have per-step losses $\ell^A_t,\ell^B_t$ on the same test period. Let $d_t=\ell^A_t-\ell^B_t$.
The test statistic is

$$
DM=\frac{\bar d}{\sqrt{\widehat{\operatorname{Var}}(\bar d)}},\qquad
\widehat{\operatorname{Var}}(\bar d)=\frac{1}{T}\Big(\gamma_0+2\sum_{k=1}^{h-1}\gamma_k\Big),
$$

with $\gamma_k$ the autocovariances of $d_t$ — the HAC correction exists because forecast errors at
horizon $h$ are autocorrelated up to lag $h-1$. Under the null of equal accuracy, $DM$ is
approximately standard normal.

What matters in practice:

* It puts a **confidence interval on the difference**, which is the only quantity you care about. A 3 % RMSE improvement with $DM=0.6$ is not an improvement.
* It is a test on **one series**. Across many series use the Wilcoxon signed-rank / Friedman + post-hoc procedure (the standard in the classification bake-offs of Chapter 4).
* It is invalid for nested models estimated on the same data, and its small-sample version (Harvey-Leybourne-Newbold) should be used when $T$ is short.
* It says nothing about whether the gap *matters* for the decision. Statistical and practical significance are separate arguments; make both.

## A protocol checklist to paste into every project

1. Unit of evaluation and unit of splitting (they are often different).
2. Split rule: rolling origin / blocked+embargo / group-by-platform — with the reason.
3. All preprocessing (scaling, imputation, feature selection, augmentation) fitted **inside** the fold.
4. Baselines from Chapter 0 evaluated on exactly the same folds.
5. Metric matched to the framing (Chapter 0 Lesson 01); horizon-wise breakdown for forecasts.
6. Uncertainty on the *difference* between methods (DM, or fold-level distributions).
7. One leakage probe: label shuffle, or a one-step shift test.

## Assumptions & failure modes

| Assumption | How it breaks | Symptom |
| --- | --- | --- |
| Test period is representative | one regime change dominates the test window | huge variance across folds; conclusions flip with the cutoff date |
| Folds are independent | overlapping windows, shared platforms | over-confident significance |
| Enough test data | short series, long horizon | any ranking is noise — say so |
| The metric is what the user cares about | RMSE reported, decisions taken at a threshold | model chosen on the wrong functional (Ch.0 L01) |

**Lens check:** lens 2, in its entirety.

## Available Challenges

[Challenge 01 - Trick-Series Diagnosis](../challenges/challenge1_trick_series.ipynb)
