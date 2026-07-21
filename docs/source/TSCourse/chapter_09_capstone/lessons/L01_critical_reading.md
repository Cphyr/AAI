```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - Reading TS/Trajectory Papers Critically

## Motivation

By now you have seen the same failure repeatedly: a result that is real on the benchmark and absent
in your data. This lesson turns the course's three lenses into a reading procedure, so that you can
predict — before spending two weeks reproducing — whether a claim will survive contact with your
problem.

## The procedure

Read in this order. It is deliberately not the paper's order.

**1. What is the claim, exactly?** Write it as one sentence with a number, a dataset and a metric.
If you cannot, the paper has not made a claim and there is nothing to reproduce.

**2. What is the task framing?** Forecasting, filtering, classification, anomaly detection
(Ch.0 L01)? Does the evaluation match that framing, or is a filtering method being evaluated with
smoothed ground truth?

**3. Baselines.** Are the naive ones present — persistence, seasonal naive, constant velocity,
1NN-DTW, a linear model, a feature+RF pipeline (Ch.0 L02, Ch.4)? Were they tuned with the same
effort as the proposed method? A paper whose baseline table starts at "LSTM" has skipped the part
that matters.

**4. Protocol and leakage.** (Ch.1 L03.) Chronological split or random? Nested hyperparameter
selection, or tuned on the test set? Grouped by platform/subject? Normalisation fitted inside the
fold? Overlapping windows across the split boundary? Is the reported number the best over many runs
(a maximum, not an estimate)?

**5. Do the gains survive statistics?** Fold-level results or a single number? Multiple seeds?
Diebold-Mariano on the loss differences, or Wilcoxon/Friedman across datasets? Is the gap larger
than the seed-to-seed variance?

**6. Ablations.** Which component actually carries the gain? If normalisation, lookback length or
data augmentation is not ablated, assume it is doing the work (Ch.7 L02).

**7. Assumptions.** Map the method back to this course: what does it assume about stationarity
(Ch.1), noise structure and unimodality (Ch.2), sampling regularity (Ch.3), invariances (Ch.4),
label availability (Ch.5), exchangeability (Ch.6), interaction structure (Ch.7)? Then ask which of
those hold in *your* setting. This mapping is the deliverable of the exercise.

**8. Reproducibility surface.** Code, data, seeds, hyperparameters, compute. Missing pieces are not
proof of anything, but they set the cost of your reproduction.

**9. Causal claims.** (Ch.6 L03.) Does the paper claim an *effect* of an intervention from
observational series? Granger causality, "adding feature $x$ improved prediction", and correlated
regimes are not identification. A causal claim needs an intervention, a natural experiment, or
untreated controls plus an explicit stability assumption — and the assumption should be stated and
tested with placebos.

## Domain-specific red flags

| Field | Red flag |
| --- | --- |
| Tracking / filtering | no consistency check (NIS/NEES); tuned $Q,R$ per scenario; ground truth from a smoother of the same data; clutter density unreported |
| Multi-target | OSPA cutoff $c$ unstated (it sets the metric's meaning); identity switches unreported; $P_D$ assumed known |
| TS classification | accuracy on imbalanced data; z-normalisation choice unstated; unequal-length handling unexplained; no critical-difference diagram |
| Forecasting | lookback tuned only for the proposed model; single chronological split; metrics on normalised scale only; no naive baseline |
| Anomaly detection | point-adjusted F1; threshold chosen on the test set; ROC-AUC under a $10^{-3}$ base rate |
| Trajectory prediction | min-ADE$_k$ only; no constant-velocity baseline; no feasibility check; random split over scenes from the same recording |
| Foundation models | "zero-shot" on datasets plausibly in the pretraining corpus; no decontamination statement |

## The exercise

Pick one paper — a tracking paper or a trajectory-classification paper — and produce two pages:

1. The claim, in one sentence with numbers.
2. The protocol, reconstructed as a diagram: what was trained on what, evaluated on what, split how.
3. A leakage audit against the Chapter 1 Lesson 03 checklist.
4. The assumption map (step 7 above), with a column for whether each holds in your domain.
5. Your prediction: will this reproduce on new data? Which single change would break it? Write it
   down *before* running the capstone — then check yourself afterwards.

## Assumptions & failure modes (of your own reading)

| Trap | Correction |
| --- | --- |
| Believing the abstract's framing | reconstruct the protocol yourself from the experimental section and the code |
| Assuming the reported baseline is a good baseline | run your own, on the same split |
| Treating a benchmark ranking as a property of methods | it is a property of methods **×** protocol **×** data |
| Dismissing a paper because it has flaws | all papers have flaws; the question is whether the flaw threatens *this* claim |
| Reproducing before predicting | write your prediction first — that is how you learn to read faster |

**Lens check:** all three, used as a checklist rather than as a lesson.

## Capstone

[Reproduce a Central Claim](../challenges/capstone_reproduction.ipynb)
