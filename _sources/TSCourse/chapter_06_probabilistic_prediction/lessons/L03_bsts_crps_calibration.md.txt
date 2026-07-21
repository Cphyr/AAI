```
Author: Cfir Hadar

Tags: Done
```
# Lesson 03 - Bayesian Structural TS, CRPS & Calibration

## Motivation

Quantile regression and conformal prediction are model-agnostic wrappers: they produce uncertainty
without saying where structure comes from. Structural time-series models take the opposite route —
write the structure down as latent states with priors, and let inference deliver the uncertainty.
Then, whichever route you take, you need a proper score and a calibration check. That is the rest
of this lesson.

## Bayesian structural time series

A BSTS model is a linear-Gaussian state-space model (Chapter 2, Lesson 01) whose state is assembled
from interpretable components:

$$
y_t=\underbrace{\mu_t}_{\text{level}}+\underbrace{\tau_t}_{\text{seasonal}}+\underbrace{\beta^\top x_t}_{\text{regression}}+\varepsilon_t,
\qquad
\begin{aligned}
\mu_{t+1}&=\mu_t+\delta_t+\eta_{\mu,t}\\
\delta_{t+1}&=\delta_t+\eta_{\delta,t}\quad(\text{slope})\\
\tau_{t+1}&=-\sum_{s=1}^{m-1}\tau_{t+1-s}+\eta_{\tau,t}
\end{aligned}
$$

The design *is* the domain assumption: a local-level model says shocks to the level are permanent;
adding a slope says trends persist; the seasonal recursion says the $m$ seasonal effects sum to
zero over a cycle; a regression block says covariates act contemporaneously. Priors on the variance
of each $\eta$ encode how quickly each component may move — a slope variance near zero is a claim
that the trend is nearly deterministic.

What this buys over a black box: components you can plot and defend separately, natural handling of
missing data (the filter simply skips the update), uncertainty from the posterior rather than
bolted on, and easy inclusion of domain covariates. `statsmodels`' `UnobservedComponents` and
`tfp.sts` are the accessible implementations. What it costs: linearity, Gaussianity, and
inference time.

**CausalImpact lineage.** Fit a BSTS to a treated series using untreated control series as
covariates, on the *pre-intervention* period; forecast the counterfactual forward; the difference
between observed and counterfactual, integrated over the post-period, is the estimated effect with
a credible interval. It is a clean and useful tool — and it is only as good as its assumption that
the controls were *not* affected by the intervention and that the pre-period relationship persists.

**On causal claims from time series.** Granger causality tests whether $x$ improves the prediction
of $y$ beyond $y$'s own past. That is *predictive precedence*, not causation: a common driver, a
slow confounder, or a series that anticipates the cause (markets, schedules, weather forecasts)
produces Granger causality with no causal link. From observational series you can claim an
intervention effect only with an explicit identification argument — an actual intervention, a
credible natural experiment, or untreated controls plus a stability assumption (as in
CausalImpact). "The model got better when I added $x$" is not one. You will revisit this when
reading papers in Chapter 9.

## CRPS: the right score for a distribution

RMSE evaluates a point. To score a full predictive distribution $F$ against the realised $y$, use
the **continuous ranked probability score**:

$$
\mathrm{CRPS}(F,y)=\int_{-\infty}^{\infty}\big(F(u)-\mathbb 1[u\ge y]\big)^2\,du
\;=\;\mathbb E|X-y|-\tfrac12\mathbb E|X-X'| ,\qquad X,X'\sim F \text{ i.i.d.}
$$

Properties that make it the default:

* **Proper**: it is minimised in expectation by the true predictive distribution, so you cannot
  game it by over- or under-dispersing.
* The second form gives a free **sample-based estimator** — for any model you can sample from
  (particle filter, MC dropout, ensemble, generative trajectory model), CRPS is computable without
  a density.
* It reduces to absolute error when $F$ is a point mass, so point and probabilistic forecasts are
  comparable on one scale.
* It decomposes into calibration and sharpness terms, and averaging pinball loss over a grid of
  $\tau$ approximates it (Lesson 01).

Alternatives: log score (very sensitive to tails — one zero-probability event dominates), energy
score and variogram score for multivariate/trajectory outputs (the energy score's power against
mis-specified correlation is weak; the variogram score is better at that). For an interval rather
than a full distribution, use the Winkler score.

**Do not use MAPE on trajectory data** — undefined near zero, asymmetric, and unit-dependent.

## Calibration diagnostics

* **PIT histogram.** Compute $u_t=F_t(y_t)$. If the forecasts are calibrated, $u_t\sim\mathrm{Unif}(0,1)$.
  A U-shaped histogram means over-confidence (too narrow); a hump means under-confidence; a slope
  means bias. This one plot diagnoses more than any table.
* **Reliability diagram** — predicted probability vs. observed frequency, for event forecasts.
* **Coverage vs. nominal curve** — for each nominal level plot empirical coverage; the ideal is the
  diagonal.
* **Rank histogram** (the ensemble version of PIT) when your forecast is a finite set of members.
* Always do all of these **conditionally**: by horizon, by regime, over rolling windows. Marginal
  calibration is easy to fake by cancelling opposite errors.

**Trajectory envelopes ("tubes").** For a predicted path, calibration is a *joint* question: what
fraction of true trajectories stay inside the whole tube for the whole horizon? That number is much
lower than per-step coverage would suggest, because a path escapes if it escapes at *any* step.
Report both: per-step coverage (per horizon) and whole-path coverage. For 2D positions, the
per-step analogue of PIT is the Mahalanobis distance under the predicted covariance, which should
follow $\chi^2_2$ — the same NEES check as Chapter 2 Lesson 01, reappearing as a calibration tool.

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| Components are additive, linear, Gaussian | multiplicative seasonality, heavy tails | systematic residual structure | log transform, or a non-Gaussian SSM |
| Priors are weakly informative | tight variance priors on a moving component | model refuses to follow real change | prior sensitivity check |
| Pre-period relationship persists (CausalImpact) | controls affected, or regime change | "effect" that is really drift | placebo tests on untreated periods/units |
| Granger causality means causation | confounding, anticipation | policy advice from a correlation | require an identification argument |
| Average calibration is calibration | regime-dependent errors | flat PIT overall, U-shaped per regime | conditional PIT/coverage |
| Per-step coverage = path coverage | multi-step tubes | tube covers far fewer whole paths | report whole-path coverage |

**Lens check:** lens 1 (structural components are an explicit temporal representation), lens 2
(CRPS and calibration are the evaluation machinery), lens 3 (prior/assumption checks).

## Walkthrough

[Three Ways to Build Intervals](../walkthroughs/lesson_three_intervals.ipynb)

## Available Challenges

[Challenge 01 - Regime-Change Coverage](../challenges/challenge1_regime_coverage.ipynb)
