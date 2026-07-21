```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - Linear-Gaussian SSMs & the Kalman Filter

## Motivation

A state-space model says: there is a hidden state that evolves by its own rules, and you see noisy
functions of it. That single idea covers tracking, navigation, sensor fusion, structural time
series, HMMs, and — read generously — recurrent networks. When the dynamics are linear and the
noise Gaussian, the exact Bayesian solution is available in closed form and costs a few matrix
multiplications per step: the Kalman filter. This lesson derives it, because the derivation is what
tells you *when it stops being correct*.

## The model

$$
\begin{aligned}
x_t&=F x_{t-1}+B u_t+w_t, & w_t&\sim\mathcal N(0,Q) &&\text{(dynamics, process noise)}\\
z_t&=H x_t+v_t, & v_t&\sim\mathcal N(0,R) &&\text{(measurement, sensor noise)}
\end{aligned}
$$

with $w,v$ independent, white, and independent of $x_0\sim\mathcal N(\hat x_0,P_0)$.

**Anatomy, mapped to a real trajectory.** For 2D constant-velocity tracking,
$x=(p_x,p_y,\dot p_x,\dot p_y)^\top$, sampling interval $\Delta t$, position-only sensor:

$$
F=\begin{pmatrix}I_2&\Delta t\,I_2\\0&I_2\end{pmatrix},\qquad
H=\begin{pmatrix}I_2&0\end{pmatrix},\qquad
Q=\sigma_a^2\,G G^\top,\quad G=\begin{pmatrix}\tfrac{\Delta t^2}{2}I_2\\ \Delta t\,I_2\end{pmatrix}.
$$

* $F$ — your physics. Constant velocity here; Lesson 03 gives the alternatives.
* $Q$ — *how wrong your physics is*. Not "noise" in a sensor sense: it is the budget for
  unmodelled accelerations. $\sigma_a$ is a real, tunable, physical quantity (how hard can this
  platform maneuver?). Too small and the filter refuses to follow turns; too large and it chases
  measurement noise.
* $R$ — sensor error covariance, from the datasheet, from calibration, or estimated. Off-diagonal
  terms encode correlated errors (e.g. range-bearing converted to Cartesian).
* $H$ — what the sensor actually sees. Stacking sensors into $H$ and block-diagonal $R$ is all
  that "sensor fusion" means here.

## Derivation as recursive Bayes

The filtering distribution is $p(x_t\mid z_{1:t})$. Two operations alternate:

$$
\underbrace{p(x_t\mid z_{1:t-1})=\int p(x_t\mid x_{t-1})\,p(x_{t-1}\mid z_{1:t-1})\,dx_{t-1}}_{\textbf{predict (Chapman-Kolmogorov)}},
\qquad
\underbrace{p(x_t\mid z_{1:t})\propto p(z_t\mid x_t)\,p(x_t\mid z_{1:t-1})}_{\textbf{update (Bayes)}} .
$$

Gaussians are closed under both linear maps and multiplication, so if the prior is Gaussian
everything stays Gaussian and we only need to track a mean and a covariance.

![The Kalman filter's predict/update cycle, with Gaussian state estimates](../../../_static/ts/wm_kalman_cycle.svg)

*The whole algorithm is this loop. Source:
[Basic concept of Kalman filtering](https://commons.wikimedia.org/wiki/File:Basic_concept_of_Kalman_filtering.svg)
by Petteri Aimonen, CC0, via Wikimedia Commons.*

**Predict.** $x_t=Fx_{t-1}+w_t$ with $x_{t-1}\sim\mathcal N(\hat x_{t-1},P_{t-1})$ gives

$$
\hat x_{t|t-1}=F\hat x_{t-1}+Bu_t,\qquad P_{t|t-1}=FP_{t-1}F^\top+Q .
$$

**Update.** Stack state and measurement — they are jointly Gaussian:

$$
\begin{pmatrix}x_t\\z_t\end{pmatrix}\Bigg|\,z_{1:t-1}\sim
\mathcal N\!\left(\begin{pmatrix}\hat x_{t|t-1}\\H\hat x_{t|t-1}\end{pmatrix},
\begin{pmatrix}P_{t|t-1}&P_{t|t-1}H^\top\\ HP_{t|t-1}&\underbrace{HP_{t|t-1}H^\top+R}_{S_t}\end{pmatrix}\right).
$$

Apply the Gaussian conditioning identity $\mathbb E[a\mid b]=\mu_a+\Sigma_{ab}\Sigma_{bb}^{-1}(b-\mu_b)$,
$\operatorname{Cov}(a\mid b)=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$:

$$
\begin{aligned}
\tilde y_t&=z_t-H\hat x_{t|t-1} &&\text{innovation}\\
S_t&=HP_{t|t-1}H^\top+R &&\text{innovation covariance}\\
K_t&=P_{t|t-1}H^\top S_t^{-1} &&\text{Kalman gain}\\
\hat x_t&=\hat x_{t|t-1}+K_t\tilde y_t,\qquad P_t=(I-K_tH)P_{t|t-1}.
\end{aligned}
$$

That is the whole filter: **conditioning a joint Gaussian, once per time step**. No optimisation,
no iteration, and under the stated assumptions it is exactly the Bayesian posterior and hence the
MMSE estimator. (In code, prefer the Joseph form
$P_t=(I-K_tH)P(I-K_tH)^\top+K_tRK_t^\top$ — it stays symmetric positive-definite under rounding.)

## Reading the gain

Scalar case: $K=\dfrac{P_{t|t-1}}{P_{t|t-1}+R}$.

* Noisy sensor ($R\gg P$): $K\to0$, ignore the measurement, coast on the model.
* Sharp sensor or uncertain prediction ($P\gg R$): $K\to1$, snap to the measurement.

The filter is an **uncertainty-weighted average of prediction and measurement**, applied
recursively. Note that $P_t,K_t$ do not depend on the data at all — they follow a deterministic
Riccati recursion and converge to a steady state. Consequence: *the honesty of the reported
uncertainty depends entirely on the honesty of $Q$ and $R$*, never on the measurements.

## Consistency diagnostics (use these, always)

The innovations $\tilde y_t$ should be zero-mean, white, with covariance $S_t$. Three checks:

* **NIS** (normalised innovation squared) $\;\tilde y_t^\top S_t^{-1}\tilde y_t\sim\chi^2_{m}$; the
  time-average should sit inside the $\chi^2$ confidence band. Too large → $Q$ or $R$ too
  optimistic, or the model is wrong; too small → you are over-inflating uncertainty.
* **NEES** (needs ground truth) $\;(x_t-\hat x_t)^\top P_t^{-1}(x_t-\hat x_t)\sim\chi^2_n$.
* **Innovation whiteness**: autocorrelated innovations mean unmodelled dynamics (a maneuver, a
  bias — see Lesson 03).

**Gating**: reject a measurement whose Mahalanobis distance exceeds a $\chi^2$ threshold. This is
both the standard outlier guard and, in Lesson 04, the first step of data association.

## Smoothing, likelihood, relatives

* **RTS smoother** — run the filter forward, then backward:
  $\hat x_{t|T}=\hat x_t+C_t(\hat x_{t+1|T}-\hat x_{t+1|t})$ with $C_t=P_tF^\top P_{t+1|t}^{-1}$.
  Post-hoc refinement: strictly tighter than filtering, and strictly unavailable in real time.
  Using smoothed states as features for a real-time model is a leak (Ch.1 L03).
* **Likelihood for free**: $\log p(z_{1:T})=-\tfrac12\sum_t\big[\log|2\pi S_t|+\tilde y_t^\top S_t^{-1}\tilde y_t\big]$.
  Maximise it to fit unknown $Q,R$ — and use it as an anomaly score in Chapter 5.
* **Orientation**: ARIMA and ETS are linear-Gaussian SSMs in disguise (state = lagged
  values / level-trend-seasonal components); that is how `statsmodels` fits them. An HMM is the
  same graphical model with a discrete state; an RNN is the same recursion with a learned
  nonlinear update and no uncertainty (Chapter 7, Chapter 8).

## Assumptions & failure modes

| Assumption | Reality on tracks | Symptom | Response |
| --- | --- | --- | --- |
| Linear dynamics & measurement | range-bearing radar, turns, geodesy | biased estimates, divergence | EKF/UKF (L02) |
| Gaussian, white, zero-mean noise | outliers, glitches, correlated errors | NIS spikes, track lost to one bad return | gating, robust/heavy-tailed models |
| Known, constant $Q,R$ | maneuvers change $Q$; sensor quality varies | NIS drifts out of band; lag during turns | adaptive $Q$, IMM (L03), ML estimation |
| Unimodal posterior | ambiguous association, multipath | confident and wrong | particle filter / MHT (L02, L04) |
| Model order is right | unmodelled bias, clock drift | autocorrelated innovations | augment the state with the bias |

**Lens check:** lens 1 (state *is* the representation), lens 2 (NIS/NEES are the uncertainty
audit), lens 3 (every row of the table above).

## Walkthrough

[Maneuvering Target with Clutter](../walkthroughs/lesson_maneuvering_target.ipynb)

## Next

[Lesson 02 - Beyond Linear-Gaussian: EKF, UKF, Particle Filters](L02_nonlinear_filters.md)
