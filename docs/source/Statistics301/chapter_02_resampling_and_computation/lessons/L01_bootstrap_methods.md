# Bootstrap Methods

## Motivation

Statistics 1/2 derived standard errors and confidence intervals from closed-form asymptotic
formulas: $\widehat{SE}(\bar X) = S/\sqrt n$, the delta method, or exact pivots like
$t=(\bar X-\mu)/(S/\sqrt n)$. These require either a known sampling distribution or a tractable
asymptotic approximation. Many estimators of practical interest -- the sample median, a trimmed
mean, a correlation coefficient, the output of a complex algorithm -- have no simple variance
formula. The bootstrap (Efron, 1979) replaces analytic derivation with computation: treat the
observed data as a stand-in for the population and resample from it repeatedly to approximate
the sampling distribution of a statistic.

## The Plug-In Principle and the Empirical Distribution

Let $X_1,\dots,X_n \stackrel{iid}\sim F$ and let $\theta=\theta(F)$ be a functional of interest
(e.g. the mean, $\theta(F)=\int x\,dF$). The **empirical distribution** puts mass $1/n$ on each
observed point, $\hat F_n(x) = \frac1n\sum_i \mathbb 1\{X_i\le x\}$; by Glivenko-Cantelli,
$\hat F_n \to F$ uniformly a.s. The **plug-in estimator** is $\hat\theta=\theta(\hat F_n)$.

The bootstrap applies plug-in one level further: we want the sampling distribution of
$\hat\theta=\theta(\hat F_n)$ under repeated draws of $X_1,\dots,X_n$ from $F$, but since $F$ is
unknown we substitute $\hat F_n$ for $F$ and study $\theta(\hat F_n^*)$ where $\hat F_n^*$ is
built from a sample drawn from $\hat F_n$. Because $\hat F_n$ is discrete on
$\{X_1,\dots,X_n\}$, sampling from it means drawing $n$ points **with replacement** from the
data. Each such **bootstrap replicate** gives $\hat\theta^*_b = \theta(X_1^*,\dots,X_n^*)$;
repeating $B$ times approximates the sampling distribution of $\hat\theta$.

**Nonparametric vs. parametric bootstrap.** Nonparametric resampling draws directly from
$\hat F_n$ (the data points), making no distributional assumption. Parametric bootstrap instead
fits a model $F_{\hat\eta}$ (e.g. $N(\hat\mu,\hat\sigma^2)$) and draws resamples iid from
$F_{\hat\eta}$ itself. Parametric bootstrap has lower variance when the model is correct but is
biased if it is wrong. $B$ (number of replicates) is a Monte Carlo tuning constant, not a
statistical one; $B\approx 1000$ suffices for SEs, $B \ge 2000$ for tail-sensitive CIs.

## Bootstrap SE, Bias, and Confidence Intervals

Given replicates $\hat\theta^*_1,\dots,\hat\theta^*_B$ with mean $\bar\theta^*$, the
**bootstrap SE** is $\widehat{SE}_{boot} = \sqrt{\frac1{B-1}\sum_b(\hat\theta_b^*-\bar\theta^*)^2}$,
requiring no analytic derivative. The **bootstrap bias estimate** is
$\widehat{\text{bias}} = \bar\theta^*-\hat\theta$, giving bias-corrected $\tilde\theta =
2\hat\theta-\bar\theta^*$ (use cautiously: can raise MSE when $B$ or $n$ is small).

Four CI constructions, in increasing sophistication:

**Normal**: $\hat\theta \pm z_{1-\alpha/2}\widehat{SE}_{boot}$ -- fixes an unknown SE formula but
not skewness/bias.

**Percentile**: use empirical quantiles of replicates directly,
$(\hat\theta^*_{(\lfloor B\alpha/2\rfloor)}, \hat\theta^*_{(\lceil B(1-\alpha/2)\rceil)})$ --
transformation-respecting, no normality needed, but uncorrected for bias/skew.

**Basic (reflection)**: $(2\hat\theta - \hat\theta^*_{(\lceil B(1-\alpha/2)\rceil)},\ 2\hat\theta -
\hat\theta^*_{(\lfloor B\alpha/2\rfloor)})$ -- reflects around $\hat\theta$; not
transformation-invariant, and corrects a different issue than percentile (they coincide only
when the bootstrap distribution is symmetric about $\hat\theta$).

**BCa (bias-corrected and accelerated)**: corrects both bias and skewness/"acceleration" (how
the SE of $\hat\theta$ changes with $\theta$), using

$$
\hat z_0 = \Phi^{-1}\!\left(\frac{\#\{\hat\theta_b^*<\hat\theta\}}{B}\right), \qquad
\hat a = \frac{\sum_i(\hat\theta_{(\cdot)}-\hat\theta_{(i)})^3}{6\left[\sum_i(\hat\theta_{(\cdot)}-\hat\theta_{(i)})^2\right]^{3/2}},
$$

where $\hat\theta_{(i)}$ is the jackknife (leave-$i$-out) value and $\hat\theta_{(\cdot)}$ their
mean. Adjusted levels: $\alpha_1=\Phi\!\left(\hat z_0+\frac{\hat z_0+z_{\alpha/2}}{1-\hat
a(\hat z_0+z_{\alpha/2})}\right)$, similarly for $\alpha_2$ with $z_{1-\alpha/2}$. When $\hat
z_0=\hat a=0$, BCa reduces to the percentile interval. BCa achieves coverage error $O(1/n)$
versus $O(1/\sqrt n)$ for normal/percentile, at the cost of the jackknife step and more
sensitivity to Monte Carlo noise.

## When the Bootstrap Fails

- **Extremes**: for $\theta=\max(X_i)$, the bootstrap is inconsistent -- the resampled max
  equals the observed max with probability $1-(1-1/n)^n\to 1-e^{-1}\approx 0.632$, an atom that
  does not vanish, unlike the continuous limiting extreme-value distribution.
- **Heavy tails**: with infinite-variance $F$, the ordinary bootstrap for the mean is
  inconsistent; the $m$-out-of-$n$ bootstrap ($m\ll n$) restores consistency.
- **Non-smooth statistics**: the sample median's bootstrap distribution converges at rate
  $n^{1/3}$, not $n^{1/2}$, so BCa's smooth-function heuristics apply only loosely.
- **Dependence**: iid resampling destroys serial/cluster dependence and understates variance.
  Remedies: **block bootstrap** (resample contiguous blocks of length $\ell$), stationary
  bootstrap (random block lengths), cluster bootstrap (resample whole clusters).
- **Small $n$**: $\hat F_n$ is simply a poor stand-in for $F$, regardless of scheme.

```python
import numpy as np
def bootstrap_ci_percentile(x, statistic, B=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    reps = np.array([statistic(x[rng.integers(0, n, n)]) for _ in range(B)])
    return np.quantile(reps, [alpha/2, 1-alpha/2]), reps
```

## Worked Example

Ten skewed observations (repair time, hours): $x=(2.1,2.5,2.6,2.8,3.0,3.1,3.4,4.0,5.5,9.8)$.
Sample median $\tilde x=3.05$; no simple closed-form SE exists. With $B=2000$ resamples,
suppose $\widehat{SE}_{boot}=0.55$ and the 5th/95th replicate percentiles are $2.60,4.60$.

- Normal: $3.05\pm 1.645(0.55)=(2.14,3.95)$.
- Percentile: $(2.60,4.60)$.
- Basic: $(2(3.05)-4.60,\ 2(3.05)-2.60)=(1.50,3.50)$.

The three disagree substantially: the bootstrap distribution of the median is right-skewed (small
skewed sample, coarse order-statistic), which the symmetric normal interval cannot reflect at
all, while percentile and basic diverge because they correct different things. With $n=10$ and a
non-smooth statistic, BCa is preferred, though even it should be trusted cautiously here.

## Exercises

### Exercise 1

For $\hat\theta=\bar X_n$ with finite variance $\sigma^2$, derive the exact conditional
bootstrap variance of $\bar X_n^*$ and its limit as $n\to\infty$; compare to $S^2/n$.

<details>
<summary>Solution</summary>

Conditional on the data, $X_1^*,\dots,X_n^*$ are iid from $\hat F_n$, which has (conditional)
variance $\hat\sigma^2=\frac1n\sum_i(X_i-\bar X_n)^2 = \frac{n-1}{n}S^2$. Since
$\bar X_n^*$ is a mean of $n$ such draws,

$$
\text{Var}^*(\bar X_n^*) = \frac{\hat\sigma^2}{n} = \frac{n-1}{n}\cdot\frac{S^2}{n}.
$$

As $n\to\infty$, $\frac{n-1}{n}\to1$, so $\text{Var}^*(\bar X_n^*)\to S^2/n$, matching the usual
estimator in the limit but slightly smaller (by factor $\frac{n-1}n$) for finite $n$ -- a known
artifact of resampling from the divide-by-$n$ distribution $\hat F_n$. This is why the bootstrap
is unnecessary for the mean (the closed form is exact and better), but it shows precisely what
"bootstrap variance" targets.

</details>

### Exercise 2

Show the nonparametric bootstrap is inconsistent for $\theta(F)=\max(X_1,\dots,X_n)$ when
$X_i\stackrel{iid}\sim U(0,1)$: compute $P^*(\max X_i^*=\max X_i)$ exactly and compare to the
continuous limit of the true sampling distribution of $n(1-X_{(n)})$.

<details>
<summary>Solution</summary>

Let $M=X_{(n)}$. A bootstrap resample's max equals $M$ iff at least one of the $n$ draws hits
the (a.s. unique) index achieving $M$; each draw misses it with probability $1-1/n$
independently, so

$$
P^*(\max X_i^*=M) = 1-\left(1-\tfrac1n\right)^n \to 1-e^{-1}\approx 0.632.
$$

This atom persists in the limit. Meanwhile $P(n(1-X_{(n)})>t) = (1-t/n)^n\to e^{-t}$, a
continuous Exponential(1) limit with no atoms anywhere. Since the bootstrap distribution has a
non-vanishing atom while the true limit is continuous, the bootstrap cannot converge to the true
sampling distribution: it is inconsistent for extreme order statistics.

</details>

### Exercise 3

For $\hat\theta=g(\bar X)$ with $g$ smooth increasing and $\text{Var}(\bar X)=\sigma^2(\mu)/n$
depending on $\mu$, explain qualitatively why BCa's acceleration $\hat a$ must be nonzero
whenever $\sigma^2(\cdot)$ is non-constant, and why the plain percentile method fails here
despite being transformation-respecting.

<details>
<summary>Solution</summary>

The percentile method is valid when some monotone $h$ gives $h(\hat\theta)\sim N(h(\theta),c^2)$
with **constant** $c$ (a variance-stabilizing transform with no bias). If
$\text{Var}(\bar X)=\sigma^2(\mu)/n$ is non-constant, then by the delta method
$\text{Var}(\hat\theta)\approx g'(\mu)^2\sigma^2(\mu)/n$ still depends on $\mu$ unless $g$
happens to be exactly the variance-stabilizing transform -- generally impossible while also
being the transform of substantive interest. So no monotone reparametrization achieves constant
variance globally, and the assumption behind the plain percentile method fails.

$\hat a$ is a first-order correction for exactly this: it estimates the rate of change of
$SE(\hat\theta)$ with $\theta$ (via the jackknife skewness formula) and shifts percentile levels
asymmetrically to compensate. When $\sigma^2(\mu)$ is constant, $\hat a=0$ and BCa collapses to
percentile -- confirming $\hat a$ exists specifically to handle non-constant variance, which the
transformation-respecting-but-not-variance-aware percentile method cannot see.

</details>

### Exercise 4

For $n=500$ autocorrelated daily returns, explain why the iid case bootstrap SE for the mean is
wrong, and how the moving block bootstrap (block length $\ell$) fixes it. Give the formula
relating $\text{Var}(\bar X_n)$ to the autocorrelations $\rho_k$.

<details>
<summary>Solution</summary>

For stationary $\{X_t\}$ with autocorrelations $\rho_k$,

$$
\text{Var}(\bar X_n) = \frac{\sigma^2}{n}\left[1+2\sum_{k=1}^{n-1}\left(1-\frac kn\right)\rho_k\right].
$$

The case bootstrap resamples individual observations independently, so resampled indices have
zero serial correlation by construction regardless of the data's true dependence -- it
implicitly targets $\sigma^2/n$ (all $\rho_k=0$). If returns have positive autocorrelation, the
true variance is understated by roughly the factor $1+2\sum_k\rho_k$, and the nominal CI
undercovers.

The **moving block bootstrap** resamples contiguous blocks of length $\ell$ and concatenates
them into a resample of length $\approx n$. Choosing $\ell\to\infty$ with $\ell/n\to0$ preserves
dependence at lags $<\ell$ while allowing enough blocks for valid resampling. Too small $\ell$
under-captures long-run variance (same problem as iid); too large $\ell$ leaves too few blocks
and inflates Monte Carlo variance of the estimate itself.

</details>
