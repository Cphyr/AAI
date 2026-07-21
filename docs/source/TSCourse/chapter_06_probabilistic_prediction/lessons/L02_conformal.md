```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Conformal Prediction for Time Series

## Motivation

Conformal prediction offers something unusual: finite-sample coverage guarantees, for any model,
without distributional assumptions. The catch is the one assumption it does make —
**exchangeability** — and time series violate it by construction. This lesson covers what conformal
gives you, what it silently loses on temporal data, and what the adaptive variants recover.

## Split conformal, in four lines

Split the data into a training set and a **calibration** set. Fit $\hat f$ on training. For each
calibration point compute a **nonconformity score**, e.g. the absolute residual $s_i=|y_i-\hat f(x_i)|$.
Let $\hat q$ be the $\lceil(n+1)(1-\alpha)\rceil/n$ empirical quantile of $\{s_i\}$. Predict

$$
C(x)=\big[\hat f(x)-\hat q,\ \hat f(x)+\hat q\big].
$$

**Theorem.** If the calibration points and the test point are exchangeable, then
$P\big(y\in C(x)\big)\ge1-\alpha$, exactly, in finite samples, for any $\hat f$ and any
distribution. That is a remarkable guarantee and it costs one held-out split.

Refinements that matter in practice:

* **Normalised / locally adaptive scores**: $s_i=|y_i-\hat f(x_i)|/\hat\sigma(x_i)$ with $\hat\sigma$
  a fitted spread model, giving input-dependent widths instead of one constant band.
* **CQR (conformalised quantile regression)**: start from the quantile model of Lesson 01 and use
  $s_i=\max\{\hat q_{\alpha/2}(x_i)-y_i,\ y_i-\hat q_{1-\alpha/2}(x_i)\}$. You keep the shape and
  heteroskedasticity of the quantile fit *and* gain the coverage guarantee. This is usually the
  right default.
* Coverage is **marginal**: averaged over the distribution of $x$. It says nothing about coverage
  for a particular regime, and conditional coverage is provably impossible to guarantee in general
  without assumptions.

## What breaks on time series

Exchangeability fails for three separate reasons, worth keeping distinct:

1. **Dependence** — residuals are autocorrelated, so the calibration quantile is estimated from
   effectively fewer independent samples than $n$. The guarantee degrades but usually gracefully.
2. **Distribution shift** — the future is not drawn from the calibration distribution. This is the
   fatal one: after a regime change the calibration quantile is simply the wrong number, and
   coverage silently drops with no warning in the training metrics.
3. **Split placement** — a random calibration split leaks the future (Ch.1 L03). The calibration
   set must be a *contiguous, recent* block preceding the test period.

![Coverage under a regime change](../../../_static/ts/coverage_regime.png)

## Adaptive and online variants

**ACI (adaptive conformal inference, Gibbs & Candès 2021).** Do not fix $\alpha$; adjust it online
from realised coverage:

$$
\alpha_{t+1}=\alpha_t+\eta\big(\alpha-\mathrm{err}_t\big),\qquad \mathrm{err}_t=\mathbb 1[y_t\notin C_t].
$$

If you have been missing too often, $\alpha_t$ shrinks and intervals widen; if you have been
over-covering, they tighten. The guarantee changes character: instead of coverage per prediction,
you get **long-run average coverage** $\frac1T\sum_t\mathrm{err}_t\to\alpha$ regardless of how the
distribution shifts — a genuinely different, and for deployment often more useful, promise. The
learning rate $\eta$ is a real trade-off: large $\eta$ adapts fast and oscillates (occasionally
producing infinite-width intervals), small $\eta$ is stable and slow. AgACI and DtACI aggregate
several $\eta$'s to sidestep the choice.

**Other variants worth knowing by name**: EnbPI (bootstrap ensembles, no data-splitting cost),
NexCP (weighting calibration points by recency, so old residuals count less), and conformal PID
control (treating coverage error as a control problem). All share the same instinct: **weight or
adapt so that recent data dominates.**

## Where conformal misleads

* **Long-run coverage is not per-period coverage.** ACI can achieve 90 % overall by covering 99 %
  in calm periods and 60 % during the storm — and the storm is when you needed it.
* **Wide is not informative.** A method can hit nominal coverage by inflating widths; always report
  width alongside coverage.
* **Marginal ≠ conditional.** Report coverage broken down by regime, horizon and platform.
* **Multi-horizon and trajectory outputs.** Conformalising each horizon separately gives per-horizon
  marginal coverage, not a guarantee for the *whole path* — for a trajectory "tube" you need joint
  (Bonferroni-corrected or max-score) constructions, which are much wider than people expect.
* This is an **active research area**; treat 2021+ results as provisional and check whether a
  method's guarantee is per-prediction, long-run, or asymptotic before quoting it.

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| Exchangeability | always, in time series | coverage below nominal, drifting | contiguous calibration block; adaptive variants |
| Calibration set represents deployment | regime change | sudden coverage collapse | ACI/NexCP; monitor rolling coverage |
| Residuals homoskedastic | maneuvers, volatility clustering | intervals too wide in calm, too narrow in storms | normalised scores or CQR |
| Marginal coverage suffices | risk concentrated in one regime | fine on average, fails where it matters | conditional coverage reporting |
| Per-horizon coverage = path coverage | multi-step trajectories | the "tube" covers far less than $1-\alpha$ | joint/Bonferroni construction |

**Lens check:** lens 2 (coverage *is* evaluation of uncertainty) and lens 3 (exchangeability is the
assumption, nonstationarity is the reality).

## Next

[Lesson 03 - Bayesian Structural TS, CRPS & Calibration](L03_bsts_crps_calibration.md)
