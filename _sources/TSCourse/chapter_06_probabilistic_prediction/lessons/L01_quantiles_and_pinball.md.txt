```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - From Point Forecasts to Distributions

## Motivation

"Where will this aircraft be in 60 seconds?" has no single correct answer, and the decisions that
consume the answer — separation, collision risk, alerting — are functions of the *tail*, not of the
mean. A point forecast throws away exactly the part of the prediction the decision needs. This
lesson is about producing and scoring distributions instead.

## The loss picks the functional

Minimising expected squared error returns the conditional mean; absolute error, the median. The
generalisation is the **pinball (quantile) loss** at level $\tau\in(0,1)$:

$$
L_\tau(y,\hat q)=\begin{cases}\tau\,(y-\hat q), & y\ge\hat q\\ (1-\tau)(\hat q-y), & y<\hat q\end{cases}
\;=\;(\tau-\mathbb 1[y<\hat q])\,(y-\hat q).
$$

Its minimiser is the true $\tau$-quantile. The asymmetry is the mechanism: at $\tau=0.9$,
under-prediction is penalised nine times as heavily as over-prediction, so the optimum sits high in
the distribution.

**Quantile regression** simply fits a model with this loss. It works with anything trainable by
gradient descent, and with gradient boosting (`LightGBM` objective `quantile`) and neural networks
out of the box. Fit several $\tau$ jointly (one output per level) to share representation; be aware
of **quantile crossing** ($\hat q_{0.9}<\hat q_{0.8}$ for some inputs) and fix it by sorting the
outputs, or by a monotone architecture.

Two things quantile regression does *not* give you: a guarantee that the intervals cover (that is
Lesson 02), and a full density (that is Lesson 03 and CRPS).

## Asymmetric and heteroskedastic uncertainty

Predictive uncertainty for tracks is rarely a symmetric blob:

* Along-track error grows with speed uncertainty; cross-track error grows with heading
  uncertainty — different rates, so the envelope is an elongated ellipse, and after a possible turn
  it is not even convex.
* Physical limits truncate one side: an aircraft cannot descend below terrain, cannot exceed
  $V_{\max}$, cannot turn tighter than its bank limit.
* Uncertainty is **state-dependent**: during a turn everything widens. A model with a single global
  $\sigma$ will be over-confident in maneuvers and over-cautious in cruise — and its *average*
  calibration will look fine, which is why you must always check calibration conditionally
  (by regime, by horizon, by speed band).

Quantile regression handles all of this naturally, since each quantile is its own function of the
features.

## Why tails, concretely

Suppose a separation rule triggers when two aircraft come within 5 NM. The relevant quantity is
$P(\text{min separation} < 5\,\text{NM})$ — a tail probability. A point forecast reduces it to a
yes/no, which is either useless (never triggers) or paralysing (always triggers). Two forecasters
with identical RMSE can differ by an order of magnitude in this probability, and the one with worse
RMSE is frequently the better one for the decision because it represents the tail honestly.

For **decisions under asymmetric cost** with under-prediction cost $c_u$ and over-prediction cost
$c_o$, the optimal point action is the quantile at $\tau^\*=\dfrac{c_u}{c_u+c_o}$ — the newsvendor
result. This is the bridge from a distributional forecast back to a single number when you are
forced to produce one: *derive it from the cost ratio*, never default to the mean.

## Evaluating quantiles

* **Coverage** — the fraction of times $y$ falls below $\hat q_\tau$ should be $\tau$. Check it
  *conditionally*: by horizon, by regime, over rolling windows. Marginal coverage can be perfect
  while every window is wrong in alternating directions.
* **Pinball loss** — the proper score for a quantile; averaged over a grid of $\tau$ it approximates
  CRPS (Lesson 03).
* **Interval width** — coverage alone is trivially achieved by predicting $(-\infty,\infty)$.
  Always report sharpness alongside coverage; the goal is *sharpness subject to calibration*.
* **Winkler / interval score** — combines width and a penalty for misses in one number, which
  makes it the right single-number summary for a $(1-\alpha)$ interval.

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| One quantile summarises the risk | multi-modal futures | the 90 % interval covers a region the target will never occupy | mixtures / multi-modal models (Ch.7 L03) |
| Quantiles are monotone in $\tau$ | independently fitted levels | crossing | joint fit, sorting, monotone nets |
| Calibration on average is enough | heteroskedasticity by regime | fine overall, badly wrong in maneuvers | conditional coverage checks |
| Training distribution = deployment | drift, new platforms | coverage degrades over time | rolling recalibration (Ch.6 L02) |
| Errors are exchangeable across horizons | error grows with $h$ | one interval width for all horizons | fit per-horizon, or model $\sigma(h)$ |

**Lens check:** lens 2 (uncertainty is the deliverable) and lens 3 (heteroskedasticity and
multi-modality as assumption failures).

## Walkthrough

[Three Ways to Build Intervals](../walkthroughs/lesson_three_intervals.ipynb)

## Next

[Lesson 02 - Conformal Prediction for Time Series](L02_conformal.md)
