# Nonparametric & Semiparametric Methods

## Motivation

The t-test, ANOVA, and linear regression from Statistics 1 and 2 rest on distributional assumptions, normal errors for exact small-sample inference, or a fixed functional form (linearity) for the mean. When these assumptions are doubtful, for example with skewed or heavy-tailed data, small samples, or a mean function whose shape is genuinely unknown, we want procedures that make weaker assumptions while sacrificing as little efficiency as possible. This lesson covers three such tools: rank-based hypothesis tests that trade a controlled amount of efficiency under normality for large robustness gains under non-normality; kernel density estimation, which estimates an entire density with no parametric family assumed; and nonparametric/semiparametric regression (local regression, smoothing splines, partial linear models), which lets the mean function be estimated flexibly while keeping some structure interpretable.

## Rank-Based Tests

### Wilcoxon signed-rank test

For paired data (or one sample) testing symmetry of the differences $d_1,\dots,d_n$ about 0, rank $|d_1|,\dots,|d_n|$ from smallest (rank 1) to largest, and let $W^+$ be the sum of ranks belonging to positive $d_i$. Under $H_0$ (distribution of $d_i$ symmetric about 0), each rank is independently equally likely to belong to a positive or negative difference, giving
$$ E[W^+] = \frac{n(n+1)}{4}, \qquad Var(W^+) = \frac{n(n+1)(2n+1)}{24}. $$
For moderate/large $n$, $Z = (W^+ - E[W^+])/\sqrt{Var(W^+)}$ is approximately standard normal.

### Mann-Whitney U / Wilcoxon rank-sum test

For two independent samples $X_1,\dots,X_{n_1}$ and $Y_1,\dots,Y_{n_2}$, testing equality of distributions against a shift alternative, define
$$ U = \sum_{i=1}^{n_1}\sum_{j=1}^{n_2} \mathbb{1}(X_i>Y_j). $$
$U$ relates to the rank-sum $R_1$ of the $X$'s in the pooled sample by $U = R_1 - n_1(n_1+1)/2$. Under $H_0$,
$$ E[U] = \frac{n_1n_2}{2}, \qquad Var(U) = \frac{n_1n_2(n_1+n_2+1)}{12}. $$

### Asymptotic relative efficiency vs. the t-test

The **Pitman asymptotic relative efficiency** compares the sample sizes two tests need for the same asymptotic power against local alternatives; $ARE(A,B)=2$ means test $B$ needs twice the sample size of $A$ for equivalent power. For the Wilcoxon signed-rank test against the (paired) t-test, for a symmetric density $f$ with variance $\sigma_f^2$,
$$ ARE(W,t) = 12\,\sigma_f^2\left(\int_{-\infty}^{\infty} f(x)^2\,dx\right)^2. $$
For $f = N(0,\sigma^2)$: $\int f^2\,dx = \frac{1}{2\sigma\sqrt\pi}$, so
$$ ARE(W,t) = 12\sigma^2\left(\frac{1}{2\sigma\sqrt\pi}\right)^2 = \frac{12}{4\pi} = \frac{3}{\pi} \approx 0.955. $$
Remarkably, under exact normality the Wilcoxon test loses only about 4.5% asymptotic efficiency relative to the (optimal, under normality) t-test. Under other distributions the comparison can favor Wilcoxon substantially: the classical table of values is Normal $3/\pi \approx 0.955$, Uniform $1.0$, Logistic $\pi^2/9\approx1.097$, Double-exponential (Laplace) $1.5$, and for the Cauchy distribution the ARE is unbounded (the t-test's efficiency collapses because sample means of Cauchy variables do not even have finite variance, while rank tests remain well-behaved). This is the standard justification for preferring rank tests when normality is doubtful: bounded downside (about 5% under normality), unbounded upside under heavy tails.

## Kernel Density Estimation

### Definition, bias, and variance

Given iid $X_1,\dots,X_n \sim f$, the kernel density estimator with bandwidth $h$ and kernel $K$ (a symmetric density) is
$$ \hat f_h(x) = \frac{1}{nh}\sum_{i=1}^n K\!\left(\frac{x-X_i}{h}\right). $$
A second-order Taylor expansion of $f$ around $x$ inside $E[\hat f_h(x)]$ gives, for small $h$,
$$ Bias\big(\hat f_h(x)\big) \approx \frac{h^2}{2} f''(x)\, \mu_2(K), \qquad \mu_2(K) = \int u^2K(u)\,du, $$
and a standard variance calculation gives
$$ Var\big(\hat f_h(x)\big) \approx \frac{f(x)}{nh}\, R(K), \qquad R(K) = \int K(u)^2\,du. $$
Bias grows with $h$ (more smoothing washes out curvature) while variance shrinks with $h$ (more data points contribute to each local average): the classic bias-variance tradeoff.

### MISE and the optimal bandwidth

The mean integrated squared error is
$$ MISE(h) = \int E\big[(\hat f_h(x)-f(x))^2\big]dx \approx \underbrace{\frac{h^4}{4}\mu_2(K)^2\int f''(x)^2\,dx}_{\text{integrated squared bias}} + \underbrace{\frac{R(K)}{nh}}_{\text{integrated variance}}. $$
Differentiating with respect to $h$ and setting to zero gives the asymptotically optimal bandwidth
$$ h_{opt} = \left[\frac{R(K)}{\mu_2(K)^2 \int f''(x)^2\,dx \cdot n}\right]^{1/5} = O(n^{-1/5}), $$
and plugging back in, $AMISE(h_{opt}) = O(n^{-4/5})$. This is slower than the parametric rate $O(n^{-1})$: nonparametric density estimation pays a price in convergence rate for not assuming a parametric family, which is the generic phenomenon behind "the curse of dimensionality" in nonparametric statistics.

## Local Regression and Smoothing Splines

### Nadaraya-Watson estimator

For regression data $(X_i,Y_i)$, $i=1,\dots,n$, with $Y_i = m(X_i)+\epsilon_i$, the Nadaraya-Watson (local constant) estimator is a kernel-weighted local average:
$$ \hat m(x) = \frac{\sum_{i=1}^n K_h(x-X_i)\,Y_i}{\sum_{i=1}^n K_h(x-X_i)}, \qquad K_h(u)=K(u/h)/h. $$
This estimator has known boundary bias (near the edges of the covariate range the kernel window is asymmetric). **Local linear regression** corrects this by fitting a weighted least-squares line locally rather than a constant:
$$ (\hat a,\hat b) = \arg\min_{a,b}\sum_{i=1}^n K_h(x-X_i)\big(Y_i-a-b(X_i-x)\big)^2, \qquad \hat m(x)=\hat a. $$
Locally weighting by a line rather than a constant cancels the leading bias term at boundaries and reduces bias in regions of curvature.

### Smoothing splines

An alternative formulation directly penalizes roughness. The smoothing spline estimator solves the penalized least-squares problem over all twice-differentiable functions $m$:
$$ \hat m = \arg\min_{m}\ \sum_{i=1}^n \big(Y_i - m(X_i)\big)^2 + \lambda \int \big(m''(t)\big)^2\,dt, $$
where $\lambda \geq 0$ is the smoothing parameter. As $\lambda\to0$, $\hat m$ interpolates the data (a natural cubic spline through every point); as $\lambda\to\infty$, the penalty forces $m''\equiv0$, i.e., $\hat m$ becomes the OLS regression line. It is a classical result that the minimizer, for any $\lambda>0$, is a **natural cubic spline** with knots at the distinct $X_i$ values. The fitted values are a linear smoother, $\hat Y = S_\lambda Y$, and the **effective degrees of freedom** are $df_\lambda = tr(S_\lambda)$, generalizing the notion of "number of parameters" to a continuously tunable amount of flexibility. $\lambda$ is typically chosen by generalized cross-validation,
$$ GCV(\lambda) = \frac{n^{-1}\sum_i (Y_i-\hat m_\lambda(X_i))^2}{\big(1-tr(S_\lambda)/n\big)^2}. $$

### Semiparametric partial linear models (brief)

Sometimes we want the interpretability of a linear effect for covariates of primary interest while flexibly adjusting for a nuisance covariate:
$$ Y_i = X_i^T\beta + g(T_i) + \epsilon_i, $$
with $g(\cdot)$ an unspecified smooth function. Robinson's (1988) double-residual estimator profiles out $g$: regress (nonparametrically, e.g., via kernel smoothing) $Y$ on $T$ and each column of $X$ on $T$, take residuals $\tilde Y_i = Y_i - \hat E[Y\mid T_i]$ and $\tilde X_i = X_i-\hat E[X\mid T_i]$, then run OLS of $\tilde Y$ on $\tilde X$:
$$ \hat\beta = \left(\sum_i \tilde X_i \tilde X_i^T\right)^{-1}\sum_i \tilde X_i \tilde Y_i. $$
Remarkably, $\hat\beta$ is $\sqrt n$-consistent and asymptotically normal for $\beta$, at the parametric rate, despite $g$ being estimated only at the slower nonparametric rate; this is the hallmark "semiparametric efficiency" phenomenon: the parameter of interest is unaffected in its rate of convergence by the presence of an infinite-dimensional nuisance parameter, provided that nuisance is estimated consistently.

## Worked Example

Paired differences (treatment minus control) for $n=6$ subjects: $d = (2,\,-1,\,3,\,4,\,-0.5,\,1.5)$.

Absolute values and ranks: $|d| = (2,1,3,4,0.5,1.5)$, sorted $0.5<1<1.5<2<3<4$, so ranks are: $d=-0.5\to$ rank 1, $d=-1\to$ rank 2, $d=1.5\to$ rank 3, $d=2\to$ rank 4, $d=3\to$ rank 5, $d=4\to$ rank 6.

Positive differences are $2, 3, 4, 1.5$ with ranks $4,5,6,3$, so
$$ W^+ = 4+5+6+3 = 18. $$
$$ E[W^+] = \frac{6\times7}{4} = 10.5, \qquad Var(W^+) = \frac{6\times7\times13}{24} = \frac{546}{24} = 22.75. $$
$$ Z = \frac{18-10.5}{\sqrt{22.75}} = \frac{7.5}{4.7697} = 1.572. $$
Two-sided p-value $\approx 2(1-\Phi(1.572)) = 2(0.058) = 0.116$: not significant at $\alpha=0.05$ despite five of six differences being positive, illustrating how the small sample size limits power even for a robust test.

```python
from scipy.stats import wilcoxon
# stat, p = wilcoxon([2, -1, 3, 4, -0.5, 1.5])
```

## Exercises

### Exercise 1

Derive $E[W^+]$ and $Var(W^+)$ under $H_0$ from first principles, using the representation $W^+ = \sum_{i=1}^n i\,\psi_i$ where $\psi_i$ is the indicator that the observation with rank $i$ (by $|d|$) has a positive sign.

<details>
<summary>Solution</summary>

Under $H_0$ (the distribution of each $d_k$ is symmetric about 0), the sign of $d_k$ is independent of $|d_k|$, and by symmetry each sign is an independent fair coin flip: $P(\text{sign}=+)=P(\text{sign}=-)=1/2$, independently across the $n$ observations, *regardless of* the ranks of the $|d_k|$'s. Consequently, if we let $\psi_i \in \{0,1\}$ denote the sign attached to the observation whose $|d|$ has rank $i$ (for $i=1,\dots,n$), the $\psi_i$ are iid $Bernoulli(1/2)$, independent of which observation got which rank. Then
$$ W^+ = \sum_{i=1}^n i\,\psi_i. $$

**Mean:**
$$ E[W^+] = \sum_{i=1}^n i\, E[\psi_i] = \sum_{i=1}^n i\cdot\frac12 = \frac12\cdot\frac{n(n+1)}{2} = \frac{n(n+1)}{4}. $$

**Variance:** since the $\psi_i$ are independent, $Var(W^+) = \sum_{i=1}^n i^2\,Var(\psi_i)$. For $\psi_i\sim Bernoulli(1/2)$, $Var(\psi_i) = \tfrac12(1-\tfrac12)=\tfrac14$. So
$$ Var(W^+) = \frac14 \sum_{i=1}^n i^2 = \frac14\cdot\frac{n(n+1)(2n+1)}{6} = \frac{n(n+1)(2n+1)}{24}, $$
using the standard identity $\sum_{i=1}^n i^2 = n(n+1)(2n+1)/6$. Both formulas match the ones quoted in the notes.

</details>

### Exercise 2

Derive $E[U]$ and $Var(U)$ for the Mann-Whitney statistic under $H_0$ (both samples drawn from the same continuous distribution), using the fact that under $H_0$ every assignment of ranks $1,\dots,n_1+n_2$ to the pooled sample, with $n_1$ of them going to the $X$ sample, is equally likely.

<details>
<summary>Solution</summary>

Let $N=n_1+n_2$. Under $H_0$, the $N$ observations are exchangeable, so the set of ranks $1,\dots,N$ assigned to the $X$-sample is a uniformly random subset of size $n_1$ out of $N$ (all $\binom{N}{n_1}$ subsets equally likely). Write $R_1 = \sum_{i=1}^{n_1} \text{rank}(X_i)$ for the rank sum of the $X$'s, and recall $U = R_1 - n_1(n_1+1)/2$.

**Mean of $R_1$:** each of the $N$ ranks $1,\dots,N$ is equally likely to be any particular rank value, and by symmetry each individual $X_i$'s rank has the same marginal distribution as a single rank drawn uniformly from $\{1,\dots,N\}$ (this follows from the exchangeability/uniform-subset argument: the marginal probability that any specific rank value $r$ is assigned to *some* $X$ is $n_1/N$, and by symmetry this is spread equally over which $X_i$ gets it). So
$$ E[R_1] = n_1 \cdot E[\text{rank}] = n_1\cdot \frac{N+1}{2}. $$
Then
$$ E[U] = E[R_1] - \frac{n_1(n_1+1)}{2} = n_1\frac{N+1}{2} - \frac{n_1(n_1+1)}{2} = \frac{n_1}{2}\big[(N+1)-(n_1+1)\big] = \frac{n_1}{2}(N-n_1) = \frac{n_1n_2}{2}, $$
using $N-n_1=n_2$.

**Variance of $R_1$:** $R_1$ is the sum of $n_1$ values drawn *without replacement* from $\{1,\dots,N\}$. For sampling without replacement, the variance of a sum of $n_1$ draws from a finite population of size $N$ with population variance $\sigma_P^2 = \frac{1}{N}\sum_{r=1}^N (r-\bar r)^2 = \frac{N^2-1}{12}$ (variance of a discrete uniform on $1,\dots,N$) is
$$ Var(R_1) = n_1\sigma_P^2\cdot\frac{N-n_1}{N-1} = n_1\cdot\frac{N^2-1}{12}\cdot\frac{n_2}{N-1} = \frac{n_1n_2(N+1)}{12}, $$
using $(N^2-1)/(N-1) = N+1$. Since $U = R_1 - \text{const}$, $Var(U)=Var(R_1)$, so
$$ Var(U) = \frac{n_1n_2(n_1+n_2+1)}{12}, $$
matching the notes.

</details>

### Exercise 3

Derive the bias formula $Bias(\hat f_h(x)) \approx \frac{h^2}{2}f''(x)\mu_2(K)$ for the kernel density estimator via a Taylor expansion, stating the regularity conditions used.

<details>
<summary>Solution</summary>

By definition, $E[\hat f_h(x)] = E\left[\frac1h K\left(\frac{x-X}{h}\right)\right] = \int \frac1h K\left(\frac{x-u}{h}\right) f(u)\, du$ (using that the $X_i$ are iid with density $f$, and linearity of expectation over the sum of $n$ identical terms divided by $n$).

Substitute $u = x - ht$ (so $t = (x-u)/h$, $du = -h\,dt$; as $u$ ranges over $\mathbb{R}$, so does $t$, with the sign flip absorbed by reversing limits):
$$ E[\hat f_h(x)] = \int K(t)\, f(x-ht)\, dt. $$
Assume $f$ is twice continuously differentiable in a neighborhood of $x$ with bounded second derivative, and $K$ is a symmetric density with $\int K(t)dt=1$, $\int tK(t)\,dt=0$ (symmetry), $\int t^2K(t)\,dt = \mu_2(K) < \infty$. Taylor-expand $f(x-ht)$ around $t=0$ (i.e., around the point $x$):
$$ f(x-ht) = f(x) - ht f'(x) + \frac{h^2t^2}{2}f''(x) + o(h^2) $$
uniformly for $t$ in the support where $K$ has appreciable mass (standard regularity: $h\to 0$ as $n\to\infty$, and $K$ has compact support or sufficiently thin tails so the remainder integrates to $o(h^2)$).

Integrate term by term against $K(t)\,dt$:
$$ E[\hat f_h(x)] = f(x)\int K(t)\,dt - hf'(x)\int tK(t)\,dt + \frac{h^2}{2}f''(x)\int t^2K(t)\,dt + o(h^2). $$
The first integral is 1, the second is 0 by symmetry of $K$, and the third is $\mu_2(K)$ by definition. So
$$ E[\hat f_h(x)] = f(x) + \frac{h^2}{2}f''(x)\mu_2(K) + o(h^2), $$
and therefore
$$ Bias(\hat f_h(x)) = E[\hat f_h(x)] - f(x) \approx \frac{h^2}{2}f''(x)\mu_2(K), $$
as claimed. The bias is largest in magnitude where $|f''(x)|$ is largest (near peaks and troughs of the density) and vanishes as $h\to0$, at rate $O(h^2)$.

</details>

### Exercise 4

Given data $(X_i,Y_i)$: $(1,2), (2,3), (3,5), (4,4), (5,6)$, compute the Nadaraya-Watson estimate $\hat m(3)$ using a standard Gaussian kernel $K(u) = \frac{1}{\sqrt{2\pi}}e^{-u^2/2}$ with bandwidth $h=1$.

<details>
<summary>Solution</summary>

Compute $u_i = (x_0-X_i)/h = (3-X_i)/1$ for $x_0=3$, and the corresponding kernel weights $K(u_i) = \frac{1}{\sqrt{2\pi}}e^{-u_i^2/2}$:

| $X_i$ | $u_i$ | $K(u_i)$ | $Y_i$ | $K(u_i)Y_i$ |
|---|---|---|---|---|
| 1 | 2 | $0.05399$ | 2 | $0.10798$ |
| 2 | 1 | $0.24197$ | 3 | $0.72591$ |
| 3 | 0 | $0.39894$ | 5 | $1.99470$ |
| 4 | -1 | $0.24197$ | 4 | $0.96788$ |
| 5 | -2 | $0.05399$ | 6 | $0.32394$ |

Sum of weights: $0.05399+0.24197+0.39894+0.24197+0.05399 = 0.99086$.

Sum of weighted $Y$: $0.10798+0.72591+1.99470+0.96788+0.32394 = 4.12041$.

$$ \hat m(3) = \frac{\sum_i K(u_i)Y_i}{\sum_i K(u_i)} = \frac{4.12041}{0.99086} = 4.158. $$

The estimate 4.158 is close to but not exactly $Y_3=5$ or the simple average of neighbors, because the Gaussian kernel gives substantial (though declining) weight to $X=2$ and $X=4$ and smaller weight to the more distant points $X=1,5$, pulling the local average down from $Y_3=5$ toward the neighboring values $3$ and $4$.

</details>
