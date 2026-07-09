```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - Classical Time Series Analysis

## Motivation

A time series $\{y_t\}$ breaks the central assumption behind everything in Chapters 1-4: samples are **not i.i.d.** — the order carries the information. This one fact changes modeling (we must model dependence over time), and, just as importantly, it changes **validation**: random train/test splits leak the future into the past. Always split by time, and cross-validate with a rolling/expanding window.

Classical (linear, statistical) methods remain the right starting point: they are strong baselines, data-efficient, interpretable, and give calibrated uncertainty. Deep methods (Lesson 03) must beat them to justify themselves — and on a single short series, they usually don't.

## Stationarity

A process is **weakly stationary** if its mean and covariance structure don't drift:

$$
\mathbb{E}[y_t]=\mu,\qquad
\text{Cov}(y_t,y_{t-h})=\gamma(h)\ \ \text{(depends only on the lag } h\text{)}.
$$

Stationarity is what makes learning from a *single realization* possible: the process repeats its statistical behavior over time, so the past is informative about the future. Real series usually aren't stationary — they have **trend** and **seasonality** — and the classical workflow is: *transform to stationarity, model what remains*.

Standard transformations: differencing $\nabla y_t=y_t-y_{t-1}$ (removes trend; seasonal differencing $y_t-y_{t-s}$ removes seasonality), and log/Box-Cox transforms (stabilize growing variance). Test for a unit root with the ADF test (null: non-stationary) / KPSS (null: stationary); in practice, difference until the ACF decays quickly. A classical alternative view is explicit decomposition $y_t=T_t+S_t+R_t$ (e.g. STL) — always *plot* the decomposition before modeling anything.

## ACF and PACF: the Diagnostic Tools

* **Autocorrelation function**: $\rho(h)=\gamma(h)/\gamma(0)$ — correlation of the series with its lag-$h$ self.
* **Partial autocorrelation** $\phi_{hh}$: correlation between $y_t$ and $y_{t-h}$ *after regressing out* lags $1,\dots,h-1$ — the "direct" effect of lag $h$.

These two plots are the fingerprint of the process and the classical model-identification tool (see the table below).

## ARMA Models

The workhorse family combines two mechanisms, driven by white noise $\varepsilon_t\sim WN(0,\sigma^2)$:

**AR(p)** — the present is a regression on the recent past:

$$
y_t=c+\phi_1y_{t-1}+\cdots+\phi_py_{t-p}+\varepsilon_t .
$$

Think "momentum/feedback"; stationary iff the roots of $1-\phi_1z-\cdots-\phi_pz^p$ lie outside the unit circle (for AR(1): $|\phi_1|<1$, giving $\rho(h)=\phi_1^{|h|}$ — geometric memory).

**MA(q)** — the present is a moving average of recent *shocks*:

$$
y_t=c+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}.
$$

Think "finite echo of disturbances": memory cuts off exactly after $q$ lags.

**ARMA(p,q)** combines both; **ARIMA(p,d,q)** is ARMA applied to the $d$-times differenced series; **SARIMA(p,d,q)(P,D,Q)$_s$** adds the same structure at the seasonal lag $s$. Identification:

| | ACF | PACF |
| --- | --- | --- |
| AR(p) | decays gradually | **cuts off after lag p** |
| MA(q) | **cuts off after lag q** | decays gradually |
| ARMA | decays | decays |

In practice, don't rely on eyeballing alone: fit a small grid of $(p,d,q)$ and select by **AIC/BIC** (`auto_arima` automates this). Then *check the residuals*: they should be white noise (Ljung-Box test, residual ACF flat). Structured residuals = information your model missed.

Forecasts come with closed-form predictive variances → honest prediction intervals that widen with horizon — one of the big practical advantages of this family.

## Exponential Smoothing (ETS)

The other classical family forecasts with exponentially decaying weights on the past. Simple exponential smoothing, $\hat y_{t+1}=\alpha y_t+(1-\alpha)\hat y_t$, extends to trend and seasonality (**Holt-Winters**):

$$
\begin{aligned}
\ell_t&=\alpha\,(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1}) && \text{(level)}\\
b_t&=\beta\,(\ell_t-\ell_{t-1})+(1-\beta)\,b_{t-1} && \text{(trend)}\\
s_t&=\gamma\,(y_t-\ell_t)+(1-\gamma)\,s_{t-m} && \text{(seasonality)}\\
\hat y_{t+h}&=\ell_t+h\,b_t+s_{t+h-m\lceil h/m\rceil} .
\end{aligned}
$$

ETS and ARIMA overlap but are not nested; ETS is often the stronger default for seasonal business data, and it is embarrassingly hard to beat on short series. (The M-competitions' recurring lesson: simple statistical methods and their combinations beat sophisticated ones far more often than expected.)

## Beyond One Series

* **Exogenous inputs**: ARIMAX/regression with ARMA errors — add known drivers (weather, promotions, holidays).
* **Multivariate**: VAR models a vector of series jointly, each variable regressed on lags of *all* variables — the right classical tool when series interact (e.g. an array of correlated sensors).
* **Volatility**: if the *variance* is what clusters over time (finance), that's GARCH territory.
* When you have *many* related series, per-series classical models stop scaling and global deep models start winning — that transition is exactly Lesson 03.

## Walkthrough

[Walkthrough - Classical Forecasting with ARIMA](../walkthroughs/lesson_classical_ts_arima.ipynb) — decomposition, ACF/PACF identification, SARIMA fitting, residual diagnostics, and rolling-origin backtesting.
