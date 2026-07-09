```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - State-Space Models and the Kalman Filter

## Motivation

ARIMA models the *observations* directly. Often the more honest picture is: there is a **hidden state** evolving over time (the true position and velocity of a vehicle, the true temperature), and we only see **noisy measurements** of it (GPS fixes with error, a cheap sensor with bias and jitter). This is the world of tracking, navigation, and sensor processing — targets moving through space, geographic tracks, arrays of imperfect sensors — and its foundational tool is the Kalman filter (1960). It is still, six decades on, the most widely deployed estimation algorithm in existence (GPS receivers, aircraft, robots, spacecraft — all run one).

## The Linear-Gaussian State-Space Model

Two equations, one for how the world evolves and one for what we measure:

$$
\begin{aligned}
x_t&=F\,x_{t-1}+w_t, &\quad w_t&\sim\mathcal{N}(0,Q) &&\text{(dynamics / process noise)}\\
z_t&=H\,x_t+v_t, &\quad v_t&\sim\mathcal{N}(0,R) &&\text{(measurement / sensor noise)}
\end{aligned}
$$

Example — tracking in 1D with a constant-velocity model, state $x_t=(p_t,\ \dot p_t)^T$, sampling interval $\Delta t$, position-only sensor:

$$
F=\begin{pmatrix}1&\Delta t\\0&1\end{pmatrix},
\qquad
H=\begin{pmatrix}1&0\end{pmatrix}.
$$

$Q$ encodes how much the true motion deviates from constant velocity (maneuvers); $R$ encodes the sensor's error statistics — this is where datasheet accuracy, GPS error models, and calibration enter. The same template covers 2D/3D geographic tracks (state = position + velocity per axis) and structural time-series models (state = level/trend/seasonal — the "unobserved components" view of Lesson 01's models; ARIMA itself can be cast in state-space form).

## The Kalman Filter

We want $p(x_t\mid z_{1:t})$ — the belief about the current state given all measurements so far. Because everything is linear-Gaussian, this posterior is Gaussian, $\mathcal{N}(\hat x_t, P_t)$, and can be updated **recursively**: no need to store past data, just the current mean and covariance. Each step has two halves.

**Predict** (push the belief through the dynamics):

$$
\hat x_{t|t-1}=F\,\hat x_{t-1},\qquad
P_{t|t-1}=F\,P_{t-1}F^T+Q .
$$

**Update** (fold in the new measurement):

$$
\begin{aligned}
\tilde y_t &= z_t-H\,\hat x_{t|t-1} &&\text{(innovation: surprise in the measurement)}\\
S_t &= H\,P_{t|t-1}H^T+R &&\text{(innovation covariance)}\\
K_t &= P_{t|t-1}H^T S_t^{-1} &&\text{(Kalman gain)}\\
\hat x_t &= \hat x_{t|t-1}+K_t\,\tilde y_t \\
P_t &= (I-K_tH)\,P_{t|t-1}.
\end{aligned}
$$

The whole algorithm is in the gain. Look at the scalar case: $K=\frac{P_{pred}}{P_{pred}+R}$, so

* noisy sensor ($R$ large) → $K\to0$: trust the model, barely move;
* confident sensor / uncertain prediction → $K\to1$: trust the measurement.

The Kalman filter is nothing but **optimally weighted averaging of prediction and measurement, with the weights given by their uncertainties**, applied recursively. Under the model assumptions it is the exact Bayesian posterior and the minimum-mean-squared-error estimator. Note that $P_t$ and $K_t$ don't depend on the data — uncertainty evolves deterministically, and honesty of the output depends entirely on honest $Q$ and $R$.

![https://upload.wikimedia.org/wikipedia/commons/a/a5/Basic_concept_of_Kalman_filtering.svg](https://upload.wikimedia.org/wikipedia/commons/a/a5/Basic_concept_of_Kalman_filtering.svg)

## Sensor Fusion and Sensor Arrays

Multiple sensors measuring the same state — GPS + odometry, an array of thermometers, radar + camera — fit with **zero new machinery**: stack the measurements into $z_t$, stack their models into $H$, and put each sensor's error covariance into a block of $R$. The filter automatically weighs each sensor by its accuracy, fuses them, and handles different sampling rates (just apply whichever sensor's update arrives; predict in between). Correlated sensor errors go in $R$'s off-diagonal blocks. A biased sensor is handled by *augmenting the state* with the bias and letting the filter estimate it — the standard trick for slowly drifting sensor errors.

Practical diagnostics: the innovations $\tilde y_t$ should be zero-mean white noise with covariance $S_t$. Systematically large innovations for one sensor → its $R$ is optimistic or it's failing; **innovation gating** (discard measurements with Mahalanobis distance $\tilde y^TS^{-1}\tilde y$ above a $\chi^2$ threshold) is the standard guard against outliers/glitches.

## Smoothing, and Life Beyond Linear-Gaussian

* **Smoothing**: offline, you also have *future* measurements; the Rauch-Tung-Striebel (RTS) smoother runs a backward pass to get $p(x_t\mid z_{1:T})$ — always tighter than the filter. Use it for post-hoc track reconstruction; use the filter for real time.
* **Nonlinear dynamics or sensors** (range-bearing radar, orientation): **EKF** linearizes around the current estimate (Jacobians); the **UKF** propagates a deterministic set of sigma points instead — better for strong nonlinearities, no Jacobians needed.
* **Non-Gaussian / multimodal posteriors** (multi-hypothesis tracking, terrain matching): **particle filters** represent the posterior by weighted samples — the general tool when Kalman assumptions genuinely break.
* Unknown $Q$/$R$ can themselves be fit by maximum likelihood: the filter yields the exact log-likelihood $\sum_t\log\mathcal{N}(\tilde y_t;0,S_t)$ — optimize it. (This is how statsmodels fits state-space models.)

The conceptual ladder to keep: **Kalman filter → EKF/UKF → particle filter**, in increasing generality and cost; and note the family resemblance to HMMs (same graphical model, discrete state) and to RNNs (a learned, nonlinear state update — the bridge to the next lesson).

## Walkthrough

[Walkthrough - Kalman Filtering: Tracking a Noisy Sensor](../walkthroughs/lesson_kalman_tracking.ipynb) — implement the filter from scratch in NumPy, track a moving object from noisy position measurements, and fuse two sensors of different quality.

## Available Challenges

[Challenge 01 - Sensor Fusion](../challenges/challenge1_sensor_fusion.ipynb)
