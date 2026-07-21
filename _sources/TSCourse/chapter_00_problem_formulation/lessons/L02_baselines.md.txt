```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Baselines & Sanity Checks

## Motivation

In time series, a nontrivial fraction of published and internal "wins" evaporate against a
one-line baseline. This is not a slur on the field: temporal data has so much structure that a
model can look excellent while learning nothing beyond "tomorrow resembles today". The defence is
cheap and non-negotiable: **implement the honest baseline first**, and only then look at your
model's number.

## The non-negotiable temporal baselines

| Baseline | Definition | The right baseline when |
| --- | --- | --- |
| **Persistence / naive** | $\hat x_{t+h}=x_t$ | series is near a random walk (prices, levels) |
| **Seasonal naive** | $\hat x_{t+h}=x_{t+h-m}$ | strong known period $m$ (daily, weekly traffic) |
| **Drift / random walk with drift** | $\hat x_{t+h}=x_t+h\cdot\frac{x_t-x_1}{t-1}$ | slow trend, no seasonality |
| **Constant velocity (CV)** | $\hat p_{t+h}=p_t+h\,\Delta t\,v_t$ | kinematic tracks — this is *the* trajectory baseline |
| **Constant turn** | extrapolate current turn rate | tracks with sustained maneuvers |
| **Climatology / marginal** | predict the unconditional mean/quantiles | the honest floor for distributional forecasts |

For classification and anomaly detection the analogues are the majority class, a stratified random
scorer, and — much stronger, and the one that embarrasses people — a small set of hand-computed
kinematic features (mean/max speed, speed histogram, turn-rate quantiles, dwell time, path
straightness $\frac{\|p_T-p_1\|}{\sum\|\Delta p\|}$, altitude profile summaries) fed to logistic
regression or a random forest. If a deep sequence model cannot beat 20 features and a random
forest, the deep model is not learning temporal structure — it is learning the marginal.

Report baselines as *skill scores* so the number is interpretable:

$$
\mathrm{Skill}=1-\frac{\mathrm{Err}_{\text{model}}}{\mathrm{Err}_{\text{baseline}}}\ ,
$$

positive means better than baseline; MASE is exactly this with the naive baseline's mean absolute
error in the denominator.

## Sanity checks that catch real bugs

* **Shuffle the target.** Retrain on permuted labels. Any skill above chance means leakage.
* **Shift by one.** Feed the model $x_{t}$ where $x_{t-1}$ was intended. A model whose accuracy *collapses* under a one-step shift and is otherwise excellent is often just reading the answer.
* **Horizon curve.** Plot error vs. horizon $h$. It must increase. A flat or non-monotone curve means the model is using information it should not have, or the metric is dominated by the marginal.
* **Sub-sample the training set.** Halve it. If nothing changes, your model is not data-limited and probably not learning much.
* **Look at the residuals in time.** Structure in residuals (autocorrelation, regime-dependent bias) is free improvement lying on the floor.

## Designing failure stories

For each baseline, write down the scenario under which it *should* break — before you run
anything. This converts baselines from a formality into a hypothesis test:

* Persistence breaks when the series mean-reverts fast relative to the horizon.
* CV extrapolation breaks during turns; error grows like $\tfrac12 a h^2$, so plot error against
  turn rate, not just its average.
* Feature + random-forest baselines break when discrimination lives in *ordering* rather than in
  marginal statistics (two classes with identical speed histograms but different sequencing).
* Seasonal naive breaks at daylight-saving boundaries, holidays, and after a regime change.

If your fancy model does not win precisely in the scenario where the baseline should break, either
the scenario is absent from your test set (fix the test set) or the model is not doing what you
think.

## Assumptions & failure modes

| Assumption | How it breaks | Symptom |
| --- | --- | --- |
| The baseline is tuned as carefully as the model | Untuned baseline, tuned model | "SOTA" that vanishes when someone tunes the baseline |
| The comparison is on identical splits and horizons | Different preprocessing per method | Ranking flips under re-implementation |
| A single test set separates the methods | Test noise larger than the gap | See Diebold-Mariano in Ch.1 L03; report uncertainty on the *difference* |

**Lens check:** lens 2 (evaluation) — and the failure stories are lens 3 (assumption-reality
mismatch) written down in advance.

## Next

Chapter 1: [Stationarity as a Modeling Convenience](../../chapter_01_ts_foundations/lessons/L01_stationarity.md)
