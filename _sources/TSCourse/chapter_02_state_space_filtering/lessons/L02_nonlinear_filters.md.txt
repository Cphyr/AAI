```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Beyond Linear-Gaussian: EKF, UKF, Particle Filters

## Motivation

Real sensors are nonlinear (a radar reports range and bearing, not $x$ and $y$), real motion is
nonlinear (turns), and real posteriors are sometimes not a single blob (which of two targets is
this? which side of the obstacle did it pass?). Each of the three filters below relaxes one of the
Kalman assumptions at a price. The skill to acquire is not implementing all three — it is knowing
which assumption your problem actually violates.

## The general problem

$$
x_t=f(x_{t-1})+w_t,\qquad z_t=h(x_t)+v_t .
$$

The Bayesian recursion (predict/update) is unchanged; what breaks is that a Gaussian pushed
through $f$ or $h$ is no longer Gaussian, so the integrals have no closed form.

![What each filter can represent](../../../_static/ts/filter_ladder.png)

## EKF — linearize

Replace $f,h$ by their first-order Taylor expansions around the current estimate; use the Jacobians
$F_t=\partial f/\partial x|_{\hat x_{t-1}}$, $H_t=\partial h/\partial x|_{\hat x_{t|t-1}}$ in the
ordinary Kalman equations.

* **Cost**: same as KF, plus Jacobians. Decades of flight-proven use (GPS/INS runs on it).
* **Assumption**: the nonlinearity is mild *over the width of your current uncertainty*. That is
  the real criterion — not "is $h$ nonlinear" but "is $h$ nearly affine across $\pm3\sigma$".
* **Failure**: strong curvature or large $P$ → the mean is biased and $P$ is optimistic, the
  filter becomes over-confident, gates the true measurement out, and **diverges**. Divergence is
  usually silent in the state estimate and loud in NIS.

Classic example: converting range-bearing $(r,\theta)$ to Cartesian. The true uncertainty region is
a banana; the EKF fits an ellipse. At long range with coarse bearing accuracy the banana is very
much not an ellipse.

## UKF — sigma points

Do not linearize the function; sample it deterministically. Choose $2n+1$ sigma points matching the
mean and covariance,

$$
\mathcal X^{(0)}=\hat x,\qquad
\mathcal X^{(i)}=\hat x\pm\big(\sqrt{(n+\lambda)P}\big)_i ,
$$

push each through the true $f$ (or $h$), and recompute a weighted mean and covariance. This is the
*unscented transform*: it captures the posterior mean and covariance to second order (third for
Gaussians) with no Jacobians.

* **Cost**: $\sim2n+1$ function evaluations per step; no derivatives — a large practical win when
  $h$ is a lookup, a map, or a chunk of legacy code.
* **Still assumes**: the posterior is adequately summarised by a mean and a covariance.
* **Failure**: multi-modality. A UKF fitted to a two-lobed posterior returns the point *between*
  the lobes — confidently, and wrongly. Also, badly scaled $\lambda$ can produce non-PSD
  covariances; the square-root formulation avoids that.

## Particle filters — sample

Represent the posterior by weighted samples $\{(x_t^{(i)},w_t^{(i)})\}_{i=1}^N$,
$p(x_t\mid z_{1:t})\approx\sum_i w_t^{(i)}\delta(x_t-x_t^{(i)})$. Sequential importance sampling:

1. **Propagate**: draw $x_t^{(i)}\sim q(x_t\mid x_{t-1}^{(i)},z_t)$; the bootstrap choice is the
   dynamics prior $p(x_t\mid x_{t-1}^{(i)})$.
2. **Weight**: $w_t^{(i)}\propto w_{t-1}^{(i)}\,p(z_t\mid x_t^{(i)})$ (for the bootstrap proposal),
   then normalise.
3. **Resample** when the effective sample size $N_{\text{eff}}=1/\sum_i (w^{(i)})^2$ drops below,
   say, $N/2$: draw $N$ particles with replacement in proportion to weight (systematic resampling
   is the standard low-variance scheme).

**Degeneracy** is the central pathology: without resampling, weight concentrates on one particle
within a few steps and the filter represents nothing. Resampling fixes that and creates
**impoverishment** — duplicated particles, lost diversity — which is why you add roughening / jitter
or use a resample-move step. Both problems worsen exponentially with state dimension; PFs are the
tool for low-dimensional states (a handful, not fifty).

* **Buys**: arbitrary $f,h$, arbitrary noise (heavy-tailed, discrete, bounded), multi-modality,
  and constraints (reject particles that leave the feasible set).
* **Costs**: $N$ times the compute, stochastic output (fix the seed), tuning, and no closed-form
  covariance.

## The decision gate

Ask, in order:

1. **Is the posterior unimodal and roughly Gaussian?** If yes, stay in the Kalman family.
2. **Is the nonlinearity mild across my current $\pm3\sigma$?** Yes → EKF. No, but still unimodal →
   UKF.
3. **Multi-modal, hard constraints, or genuinely non-Gaussian noise?** → particle filter, or a
   Gaussian *mixture* filter (a bank of Kalman filters with weights — often cheaper and enough).
4. **Multi-modality caused by *association* ambiguity rather than by dynamics?** → that is Lesson
   04's problem, and the right answer is a data-association method, not a fancier filter.

A caution worth internalising: complexity does not buy accuracy. A well-tuned EKF beats a badly
tuned particle filter almost always, and $Q$/$R$ tuning beats an algorithm upgrade more often than
anyone admits.

## Assumptions & failure modes

| Filter | Assumes | Breaks when | Diagnostic |
| --- | --- | --- | --- |
| EKF | near-affine over $\pm3\sigma$ | strong curvature, large uncertainty, initialisation far off | NIS drift, covariance collapse, divergence |
| UKF | posterior ≈ its first two moments | multi-modality; poorly scaled sigma points | estimate settles between modes; non-PSD $P$ |
| PF | enough particles to cover the posterior | high dimension, sharp likelihoods, degeneracy | $N_{\text{eff}}$ collapse; results vary with the seed |
| All | $Q,R$ approximately right | maneuvers, sensor drift | NIS out of band (L01) |

**Lens check:** lens 3 — this entire lesson is a taxonomy of assumption-reality mismatch — with
lens 2 in the diagnostics column.

## Walkthrough

[Maneuvering Target with Clutter](../walkthroughs/lesson_maneuvering_target.ipynb) — EKF vs. UKF vs.
PF vs. IMM on one scenario, with tracking quality plotted against compute.

## Next

[Lesson 03 - Motion Models & Maneuvering Targets](L03_motion_models_imm.md)
