# Likelihood Theory & Asymptotics

## Motivation

Statistics 1 and 2 gave you MLE as a recipe: write the likelihood, differentiate, solve. This
lesson makes precise *why* the MLE behaves well -- why it is consistent, asymptotically normal,
and why its variance formula is the best possible for a wide class of estimators. This machinery
underlies the EM algorithm (Lesson 2), Bayesian asymptotics (Lesson 3, via the Bernstein-von Mises
theorem), and the choice between Wald, score, and likelihood-ratio tests used throughout applied
work. Throughout, $X_1,\dots,X_n$ are iid with density $f(x;\theta)$, $\theta\in\Theta\subseteq
\mathbb R$ the true parameter, under standard regularity conditions (support of $f$ independent of
$\theta$, twice-differentiable log-density, interchangeable differentiation/integration).

## Score and Fisher Information

The **score** of a single observation is $U(\theta;x) = \partial \log f(x;\theta)/\partial\theta$.
Since $\int f(x;\theta)\,dx=1$, differentiating under the integral sign gives $E_\theta[U(\theta;X)]
= 0$. The **Fisher information** is the variance of the score,

$$
I_1(\theta) = \mathrm{Var}_\theta(U(\theta;X)) = -E_\theta\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\right],
$$

the second equality holding under regularity conditions. For an iid sample, information adds:
$I_n(\theta) = nI_1(\theta)$. Large $I$ means a sharply peaked log-likelihood, so the data pin down
$\theta$ tightly.

## The Cramer-Rao Lower Bound

**Theorem.** For any unbiased estimator $\hat\theta$ of $\theta$, $\mathrm{Var}_\theta(\hat\theta)
\geq 1/I_n(\theta)$.

**Proof sketch.** Differentiating $E_\theta[\hat\theta]=\theta$ under the integral sign gives
$\mathrm{Cov}_\theta(\hat\theta,U) = 1$ (using $E_\theta[U]=0$). Cauchy-Schwarz then gives $1 =
\mathrm{Cov}(\hat\theta,U)^2 \le \mathrm{Var}(\hat\theta)\,\mathrm{Var}(U) =
\mathrm{Var}(\hat\theta)\,I_n(\theta)$, which rearranges to the bound. Equality holds iff
$\hat\theta-\theta$ is proportional to $U$ almost surely; such estimators are called
**efficient**. $\blacksquare$

## MLE Consistency and Asymptotic Normality

**Consistency (sketch).** By the LLN, $\frac1n\ell_n(\theta) \to E_{\theta_0}[\log f(X;\theta)]$
for each fixed $\theta$. Since $E_{\theta_0}[\log f(X;\theta_0)] - E_{\theta_0}[\log f(X;\theta)] =
D_{KL}(f_{\theta_0}\|f_\theta) \geq 0$ (Jensen), the population log-likelihood is uniquely
maximized at $\theta_0$, and under uniform-convergence conditions $\hat\theta_n \to \theta_0$ in
probability.

**Asymptotic normality.** Since $U_n(\hat\theta_n)=0$, a Taylor expansion of $U_n$ around
$\theta_0$ gives $0 \approx U_n(\theta_0) + U_n'(\theta_0)(\hat\theta_n-\theta_0)$, so

$$
\sqrt n(\hat\theta_n-\theta_0) \approx \frac{U_n(\theta_0)/\sqrt n}{-U_n'(\theta_0)/n}.
$$

The numerator is a sum of $n$ iid mean-zero terms with variance $I_1(\theta_0)$, so by the CLT it
is asymptotically $N(0,I_1(\theta_0))$; the denominator converges in probability to $I_1(\theta_0)$
by the LLN. By Slutsky,

$$
\sqrt n(\hat\theta_n-\theta_0) \xrightarrow{d} N(0,\ I_1(\theta_0)^{-1}),
$$

i.e. the MLE is asymptotically efficient, attaining the Cramer-Rao bound in the limit.

## The Delta Method

If $\sqrt n(\hat\theta_n-\theta_0)\xrightarrow{d}N(0,\sigma^2)$ and $g$ is differentiable at
$\theta_0$ with $g'(\theta_0)\neq0$, Taylor-expanding $g(\hat\theta_n)$ around $\theta_0$ and
discarding the $o_P(1)$ remainder gives

$$
\sqrt n\big(g(\hat\theta_n)-g(\theta_0)\big) \xrightarrow{d} N\big(0,\ g'(\theta_0)^2\sigma^2\big).
$$

This lets you build confidence intervals for any smooth reparametrization $g(\theta)$ (odds
ratios, variance-stabilized statistics) without rederiving likelihood theory from scratch.

## Wald, Score, and Likelihood-Ratio Tests

Testing $H_0:\theta=\theta_0$, three statistics are all asymptotically $\chi^2_1$ under $H_0$:

$$
W = (\hat\theta_n-\theta_0)^2 I_n(\hat\theta_n), \qquad
S = \frac{U_n(\theta_0)^2}{I_n(\theta_0)}, \qquad
\Lambda = 2[\ell_n(\hat\theta_n)-\ell_n(\theta_0)].
$$

$W$ is the familiar $z$/$t$-statistic squared; $S$ needs no MLE under $H_1$; $\Lambda$ compares
maximized likelihoods. Taylor-expanding $\ell_n(\theta_0)$ around $\hat\theta_n$ to second order
(the linear term vanishes since $U_n(\hat\theta_n)=0$, and $-\ell_n''(\hat\theta_n)\approx
I_n(\hat\theta_n)$) gives $\Lambda \approx W$; a parallel expansion of $U_n(\theta_0)$ gives $S
\approx W$. All three agree to leading order in $n$ and differ only in finite samples.

## Worked Example

$X_1,\dots,X_n\sim\mathrm{Exponential}(\lambda)$: $\ell(\lambda)=n\log\lambda-\lambda\sum x_i$,
MLE $\hat\lambda=1/\bar X$, and $I_1(\lambda)=1/\lambda^2$ so $I_n(\lambda)=n/\lambda^2$.

With $n=25$, $\bar x=2.0$: $\hat\lambda=0.5$, $\widehat{\mathrm{Var}}(\hat\lambda) =
\hat\lambda^2/n = 0.01$, $\mathrm{SE}=0.1$. A 95% Wald CI: $0.5\pm1.96(0.1) = (0.304,0.696)$.

Testing $H_0:\lambda=1$: $W=(0.5-1)^2(25/0.25)=25.0$; $S = (25-50)^2/25 = 25.0$; $\Lambda =
2\{[25\log0.5-25]-[0-50]\} = 2(7.67)=15.34$. Here $W=S$ exactly (exponential-family feature), while
$\Lambda$ differs noticeably since $\hat\lambda$ is far from $\theta_0$ (a two-fold gap).

```python
import numpy as np
n, xbar = 25, 2.0
lam_hat = 1 / xbar
wald = (lam_hat - 1)**2 * (n / lam_hat**2)
score_stat = (n / 1 - n * xbar)**2 / (n / 1**2)
loglik = lambda lam: n * np.log(lam) - lam * n * xbar
lr = 2 * (loglik(lam_hat) - loglik(1))
print(wald, score_stat, lr)  # 25.0 25.0 15.34...
```

## Exercises

### Exercise 1

Let $X_1,\dots,X_n\sim\mathrm{Bernoulli}(p)$. Derive $I_1(p)$ two ways: (a) as the variance of the
score, (b) as minus the expected second derivative of the log-likelihood. Verify the Cramer-Rao
bound for $I_n(p)$ matches the exact variance of $\hat p=\bar X$.

<details>
<summary>Solution</summary>

Log-density: $\log f(x;p) = x\log p + (1-x)\log(1-p)$.

**(a)** Score: $U(p;x) = x/p - (1-x)/(1-p) = (x-p)/[p(1-p)]$. Since $\mathrm{Var}(X)=p(1-p)$,
$I_1(p) = \mathrm{Var}(U) = p(1-p)/[p(1-p)]^2 = 1/[p(1-p)]$.

**(b)** $\partial U/\partial p = -x/p^2 - (1-x)/(1-p)^2$. Taking $-E[\cdot]$ with $E[X]=p$:
$I_1(p) = p/p^2 + (1-p)/(1-p)^2 = 1/p+1/(1-p) = 1/[p(1-p)]$. Both agree.

**CR check.** $I_n(p)=n/[p(1-p)]$, so the bound is $\mathrm{Var}(\hat p)\geq p(1-p)/n$. But
$\mathrm{Var}(\bar X) = \mathrm{Var}(X)/n = p(1-p)/n$ exactly: the bound is attained with equality,
so $\bar X$ is an efficient estimator of $p$.

</details>

### Exercise 2

Prove the Cramer-Rao inequality from first principles for a scalar $\theta$: given unbiased
$\hat\theta$ and score $U(\theta;x)$, show $\mathrm{Var}(\hat\theta)\geq 1/I(\theta)$. State which
regularity condition is used and where.

<details>
<summary>Solution</summary>

Differentiate $E_\theta[\hat\theta] = \int\hat\theta(x)f(x;\theta)\,dx = \theta$ w.r.t. $\theta$.
The right side gives $1$. The left side requires differentiating under the integral sign (the
regularity condition: support of $f$ independent of $\theta$, dominated-convergence-type
conditions):

$$
1 = \int \hat\theta(x)\frac{\partial f}{\partial\theta}\,dx = \int \hat\theta(x)U(\theta;x)f(x;\theta)\,dx = E_\theta[\hat\theta\, U].
$$

Since $E_\theta[U]=0$, $E_\theta[\hat\theta\,U]=\mathrm{Cov}_\theta(\hat\theta,U)=1$.
Cauchy-Schwarz: $\mathrm{Cov}(\hat\theta,U)^2 \le \mathrm{Var}(\hat\theta)\mathrm{Var}(U)$, so
$1 \le \mathrm{Var}(\hat\theta)I(\theta)$, giving $\mathrm{Var}(\hat\theta)\geq 1/I(\theta)$.
Equality holds iff $\hat\theta(x)-\theta = c(\theta)U(\theta;x)$ a.s., characterizing efficient
estimators (typically in exponential families where the sufficient statistic is affine in the
score).

</details>

### Exercise 3

For $X_1,\dots,X_n\sim\mathrm{Poisson}(\lambda)$, $\sqrt n(\hat\lambda-\lambda)\xrightarrow{d}
N(0,\lambda)$ since $I_1(\lambda)=1/\lambda$. Use the delta method to find the asymptotic
distribution of $\sqrt{\hat\lambda}$ and show it stabilizes the variance.

<details>
<summary>Solution</summary>

With $g(\lambda)=\sqrt\lambda$, $g'(\lambda) = 1/(2\sqrt\lambda)$, so $g'(\lambda)^2\lambda =
\lambda/(4\lambda) = 1/4$. By the delta method,

$$
\sqrt n\big(\sqrt{\hat\lambda}-\sqrt\lambda\big) \xrightarrow{d} N(0, 1/4),
$$

a variance of $1/4$ independent of $\lambda$. This is the classical square-root variance
stabilizing transform for Poisson counts: fixed-width intervals $\pm1.96/(2\sqrt n)$ apply
uniformly on the square-root scale, whereas the raw-scale interval width $1.96\sqrt{\lambda/n}$
grows with $\lambda$. In general, solving $g'(\lambda)=1/\sqrt{\sigma^2(\lambda)}$ gives the
variance-stabilizing transform for any model.

</details>

### Exercise 4

Show by second-order Taylor expansion that $\Lambda - W = o_P(1)$ under $H_0:\theta=\theta_0$,
assuming $\ell_n$ has bounded third derivative near $\theta_0$.

<details>
<summary>Solution</summary>

Expand $\ell_n(\theta_0)$ around $\hat\theta_n$ to third order:

$$
\ell_n(\theta_0) = \ell_n(\hat\theta_n) + \ell_n'(\hat\theta_n)(\theta_0-\hat\theta_n) + \tfrac12\ell_n''(\hat\theta_n)(\theta_0-\hat\theta_n)^2 + \tfrac16\ell_n'''(\tilde\theta)(\theta_0-\hat\theta_n)^3.
$$

Since $\ell_n'(\hat\theta_n)=0$ (MLE first-order condition), the linear term vanishes, so

$$
\Lambda = 2[\ell_n(\hat\theta_n)-\ell_n(\theta_0)] = -\ell_n''(\hat\theta_n)(\hat\theta_n-\theta_0)^2 - \tfrac13\ell_n'''(\tilde\theta)(\hat\theta_n-\theta_0)^3.
$$

Since $\hat\theta_n-\theta_0=O_P(n^{-1/2})$, the cubic term is $(\hat\theta_n-\theta_0)^3 =
O_P(n^{-3/2})$, and with bounded third derivative $\ell_n'''(\tilde\theta) = O_P(n)$ (sum of $n$
roughly-iid bounded terms), so this term is $O_P(n)\cdot O_P(n^{-3/2}) = O_P(n^{-1/2})\to 0$.

For the quadratic term, define $I_n(\hat\theta_n) = -\ell_n''(\hat\theta_n)$; since $-\ell_n''
(\theta)/n \to I_1(\theta)$ uniformly near $\theta_0$ and $\hat\theta_n\to\theta_0$, this matches
the Wald statistic's information term up to $o_P(1)$ relative error. So

$$
\Lambda = I_n(\hat\theta_n)(\hat\theta_n-\theta_0)^2 + o_P(1) = W + o_P(1),
$$

i.e. $\Lambda - W \xrightarrow{P} 0$. (An analogous expansion of $U_n(\theta_0)$ around
$\hat\theta_n$ shows $S=W+o_P(1)$ too, giving the full three-way equivalence.)

</details>
