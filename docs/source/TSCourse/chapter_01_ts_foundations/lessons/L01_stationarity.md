```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - Stationarity as a Modeling Convenience

## Motivation

Stationarity is not a property the world owes you. It is an assumption that makes estimation
possible: if the joint distribution of the series is invariant to time shifts, then averaging over
*time* estimates something meaningful about the *distribution*, and one realisation is enough to
learn from. Every method that pools statistics across time — ACF, ARIMA, most deep forecasters,
almost every benchmark — leans on it somewhere. The practical skill is not testing for it; it is
knowing which of your pipeline's steps silently assumed it.

## Definitions, in the version you actually use

**Strict stationarity**: $(x_{t_1},\dots,x_{t_k})\overset{d}{=}(x_{t_1+h},\dots,x_{t_k+h})$ for all
shifts. Too strong to check. **Weak (covariance) stationarity** is the working definition:

$$
\mathbb{E}[x_t]=\mu\ \ \forall t,\qquad
\operatorname{Cov}(x_t,x_{t+h})=\gamma(h)\ \text{ depends only on } h .
$$

That is all most theory needs: a constant mean and an autocovariance that depends on lag, not on
time.

![Random walk, local level, differenced](../../../_static/ts/stationarity.png)

Two series that look alike and behave completely differently:

* **Random walk** $x_t=x_{t-1}+\varepsilon_t$: $\operatorname{Var}(x_t)=t\sigma^2$ — non-stationary, unbounded, shocks are permanent. A *unit root*.
* **Local level** $\mu_t=\phi\mu_{t-1}+\eta_t,\ x_t=\mu_t+\varepsilon_t$ with $|\phi|<1$: stationary, shocks decay geometrically, mean-reverting.

The difference is invisible to the eye on short samples and decisive for modeling: with a unit root
the best forecast is the last value; with strong mean reversion it is the mean.

## The three ways to buy stationarity

| Transform | Model implied | Use when | Cost |
| --- | --- | --- | --- |
| **Differencing** $\nabla x_t = x_t-x_{t-1}$ | stochastic trend, permanent shocks | unit root present (a random walk) | destroys level information; over-differencing injects an MA(1) with a negative root and inflates high-frequency noise |
| **Detrending** (subtract fitted trend) | deterministic trend, temporary shocks | trend is genuinely a function of time | wrong trend spec leaves structured residuals; extrapolates absurdly |
| **Explicit regime / state modeling** | trend is a latent state or a switching regime | breaks, maneuvers, mode changes | more parameters, harder to fit — but honest (Chapter 2) |

The choice is a modeling claim about **whether shocks are permanent**, not a preprocessing detail.
Differencing a trend-stationary series and detrending a unit-root series are both wrong, and both
common.

**When differencing destroys signal.** For trajectories the level *is* the signal: differencing
positions gives velocity, which discards where the target actually is — fine for a velocity model,
fatal for geofencing. Differencing also whitens exactly the low-frequency structure that
long-horizon forecasts live on, and it amplifies measurement noise (the difference of two noisy
positions has $2\sigma^2$ noise variance and a strongly negative lag-1 correlation, which then gets
misread as real dynamics).

## Where the assumption sneaks in

* **ACF/PACF plots** assume the mean and covariance are time-invariant; on a trending series the ACF decays slowly and tells you almost nothing (Lesson 02).
* **ARIMA/ETS**: the "I" *is* the assumption, chosen by you.
* **Feature standardisation** computed on the whole series (global mean/std) assumes a stable distribution — and leaks the future into the past.
* **Deep forecasters**: most benchmarks normalise per-window (RevIN-style instance normalisation) precisely because the series are not stationary; when a paper reports gains, check whether the gain came from the architecture or from the normalisation (Chapter 7 Lesson 02).
* **Any model trained once and deployed forever** assumes stationarity of the *data-generating process*, which is the assumption that actually breaks in production.

## Tests — as tools, not as ritual

**ADF** tests $H_0$: unit root. **KPSS** tests $H_0$: stationarity. They disagree usefully:

| ADF | KPSS | Reading |
| --- | --- | --- |
| reject | fail to reject | stationary |
| fail to reject | reject | unit root — difference it |
| fail to reject | fail to reject | not enough data to tell |
| reject | reject | probably a structural break or heteroskedasticity, not a unit root |

Both have low power against near-unit roots and are badly fooled by level shifts (a broken-trend
series often "tests" as a unit root). Treat the outcome as a prior, not a verdict; plot the series,
plot rolling mean/variance, and prefer a model that represents the nonstationarity explicitly.

## Assumptions & failure modes

| Assumption | How it breaks | What you see | Detection |
| --- | --- | --- | --- |
| Mean constant over time | trend, regime shift, sensor recalibration | forecasts biased with a slowly varying sign | rolling mean; residual plots over time |
| Variance constant | volatility clustering, changing sensor noise | intervals too narrow in some periods, too wide in others | rolling std; coverage by period (Ch.6) |
| Autocovariance depends on lag only | seasonality that changes shape; maneuver vs. cruise | ACF differs between segments | ACF per segment |
| Differencing fixed it | over-differencing | strongly negative lag-1 ACF in residuals | look at the ACF *after* transforming |

**Lens check:** lens 1 (representation — what transform encodes what belief) and lens 3
(assumption-reality mismatch, the whole lesson).

## Next

[Lesson 02 - ACF/PACF & the Spectral View as Diagnostics](L02_acf_and_spectrum.md)
