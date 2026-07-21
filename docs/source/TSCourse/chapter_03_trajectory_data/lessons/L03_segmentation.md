```
Author: Cfir Hadar

Tags: Done
```
# Lesson 03 - Segmentation of Long Recordings

## Motivation

A 9-hour recording is not one sample. It is a taxi, a climb, a cruise with three course changes, a
descent, and a landing — and treating it as a single object destroys everything a classifier or
anomaly detector could have used. Segmentation is how you turn a long recording into units that are
homogeneous enough to model. It is also, silently, the decision that fixes your sample size,
your label granularity, and your splitting unit for evaluation.

![Change-point segmentation](../../../_static/ts/segmentation.png)

## Two families

**Rule-based.** Thresholds on physically meaningful quantities: a trip ends after $>5$ min below
2 m/s; a maneuver starts when $|\omega|>0.02$ rad/s for $>3$ s; a flight phase changes when the
climb rate crosses a band. Cheap, auditable, easy to explain to a domain expert — and that last
property is why rule-based segmentation survives in production far more often than papers suggest.
Its weakness is brittleness: every threshold is a hyperparameter tuned on data you have seen, and
hysteresis (separate on/off thresholds, plus a minimum duration) is required or you get thousands
of one-sample segments from noise.

**Change-point detection.** Formalise the same question statistically: find times where the
distribution of a signal changes.

## PELT, in enough detail to use it

Given a signal $y_{1:T}$ and a cost $\mathcal C$ (e.g. $\mathcal C(y_{a:b})=\sum_{t=a}^{b}(y_t-\bar y_{a:b})^2$
for changes in mean; use a Gaussian log-likelihood cost for changes in mean *and* variance, or a
kernel cost for arbitrary distributional changes), solve

$$
\min_{K,\;\tau_1<\dots<\tau_K}\ \sum_{k=0}^{K}\mathcal C\big(y_{\tau_k:\tau_{k+1}}\big)\;+\;\beta K .
$$

The penalty $\beta$ per change point is what keeps the answer from being "a segment per sample";
BIC-style choices set $\beta = p\log T$ with $p$ the parameters per segment. Dynamic programming
solves this exactly in $O(T^2)$; **PELT** adds a pruning rule that discards candidate change points
which can never be optimal, giving $O(T)$ in practice under mild conditions. Practical notes:

* Run it on a *derived* signal that makes the change visible — turn rate, speed, climb rate — not
  on raw position.
* Standardise the signal first, otherwise $\beta$ has no interpretable scale.
* Sweep $\beta$ and look at the segmentation, rather than trusting a single automatic value; the
  number of segments vs. $\beta$ curve usually has an elbow that matches domain intuition.
* Enforce a minimum segment length (`min_size`) — it is the statistical version of hysteresis.
* PELT is **offline**: it sees the whole recording. Using its output as a feature for a real-time
  model is a leak (Ch.1 L03). The online counterpart is **BOCPD** (Bayesian online change-point
  detection), which maintains a distribution over "run length since the last change" and updates
  it causally — reach for it when the decision must be made in real time.

`ruptures` implements PELT, binary segmentation, and window-based methods behind one API; the
value of knowing the objective above is that you can then choose the cost function and $\beta$ on
purpose.

## Segment representation constrains what comes next

The moment you commit to a segmentation, you commit to a downstream model class:

* **Fixed-length windows** (overlapping, e.g. 30 s at 50 % overlap): the only choice most deep
  classifiers accept; overlap inflates the apparent sample size and *guarantees* leakage unless
  windows are split into blocks with an embargo (Ch.1 L03).
* **Variable-length semantic segments** (one maneuver each): matches how a domain expert labels,
  needs models that handle ragged input (DTW, feature vectors, RNNs with masking).
* **Summary vector per segment**: turns the problem into ordinary tabular ML, at the cost of all
  within-segment ordering. Often the strongest baseline (Ch.0 L02).
* **Sequence of segment labels**: makes the track a short string, which opens up Markov-chain and
  HMM models — the basis of the likelihood-based anomaly detection in Chapter 5.

Two further cautions. Segment boundaries derived using *labels* are leakage. And when the
evaluation unit is a segment but segments come from the same recording, the splitting unit must
still be the recording (or the platform) — otherwise neighbouring segments of the same flight sit
on both sides of the split.

*(Deliberately out of scope: road-network / map-matched representations — that is ground-vehicle
territory and a different toolbox.)*

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| Segments are homogeneous | gradual transitions, blended maneuvers | boundaries wander; features average two behaviours | shorter segments, or model transitions explicitly (IMM, Ch.2 L03) |
| One change type | mean *and* variance both change | mean-cost PELT misses variance-only changes | choose the cost to match the change you care about |
| $\beta$ is a technicality | it is the number-of-segments knob | over/under-segmentation, silently | sweep $\beta$; sanity-check segment-count distribution |
| Offline detection is fine | model must run in real time | look-ahead leakage | BOCPD or causal rules |
| Windows are independent samples | overlap, same recording | inflated accuracy, over-confident CIs | block splits with embargo; group by recording |

**Lens check:** lens 1 (segments as the unit of representation) and lens 2 (segmentation silently
sets your evaluation unit).

## Walkthrough

[Raw Messy Trajectory to Clean Feature-Rich Representation](../walkthroughs/lesson_trajectory_pipeline.ipynb)
