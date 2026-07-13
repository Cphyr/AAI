# Bayesian Inference & MCMC

## Motivation

Statistics 1 and 2 treated $\theta$ as a fixed unknown constant. The Bayesian paradigm instead
treats $\theta$ as random with prior $p(\theta)$, updates to a posterior $p(\theta\mid x)$ via
Bayes' rule, and reports the whole posterior as the inferential answer. For simple conjugate
models the posterior is closed-form, but realistic models (hierarchical models, the mixtures of
Lesson 2) have an intractable normalizing integral. Markov Chain Monte Carlo (MCMC) draws samples
from the posterior *without* computing that integral, which is what makes Bayesian computation
practical at scale. This lesson also connects to Lesson 1: under regularity conditions, the
posterior itself becomes asymptotically normal around the MLE as $n\to\infty$ (Bernstein-von
Mises), so frequentist and Bayesian asymptotics agree even though their finite-sample philosophies
differ.

## Posterior and Conjugacy

Bayes' rule: $p(\theta\mid x) \propto f(x\mid\theta)\,p(\theta)$ (posterior $\propto$ likelihood
$\times$ prior). A prior is **conjugate** to a likelihood if the posterior stays in the same
family.

**Beta-Binomial.** $X\mid p\sim\mathrm{Binomial}(n,p)$, $p\sim\mathrm{Beta}(a,b)$:

$$
p(p\mid x) \propto p^{x+a-1}(1-p)^{n-x+b-1} \quad\Rightarrow\quad p\mid x \sim \mathrm{Beta}(a+x,\,b+n-x).
$$

Prior pseudo-counts $a,b$ add to observed counts $x,n-x$; the posterior mean $\frac{a+x}{a+b+n}$
is a weighted average of the prior mean and $x/n$, converging to the MLE as $n\to\infty$.

**Normal-Normal (known variance).** $X_i\mid\mu\sim N(\mu,\sigma^2)$ iid, $\mu\sim
N(\mu_0,\tau_0^2)$: completing the square gives a Gaussian posterior with precisions adding,

$$
\mu\mid x \sim N(\mu_n,\tau_n^2), \qquad \frac{1}{\tau_n^2}=\frac{1}{\tau_0^2}+\frac{n}{\sigma^2}, \qquad \mu_n = \tau_n^2\left(\frac{\mu_0}{\tau_0^2}+\frac{n\bar x}{\sigma^2}\right).
$$

As $n\to\infty$, $\tau_n^2\to0$ and $\mu_n\to\bar x$: the posterior concentrates on the MLE,
matching Lesson 1's asymptotic normality with $\sigma^2/n$ playing the role of $I_n(\theta)^{-1}$.

## Credible Intervals versus Confidence Intervals

A 95% **credible interval** $[L,U]$ satisfies $P(\theta\in[L,U]\mid x)=0.95$: a direct probability
statement about $\theta$ given the *observed* data. A 95% **confidence interval** is a random
interval such that, *before* data are observed, $P(\theta\in[\hat L(X),\hat U(X)]) = 0.95$ for
every fixed $\theta$: a statement about the procedure's long-run behavior, not about $\theta$
given the data in hand. These coincide numerically in simple conjugate models with flat priors
(Normal-Normal with a flat prior recovers the Wald interval exactly), but can diverge sharply with
informative priors or constrained parameter spaces. The claim "there is a 95% chance $\theta$ is
in the interval" is only strictly licensed for the credible interval.

## Metropolis-Hastings and Gibbs Sampling

**Metropolis-Hastings.** Propose $\theta^*\sim q(\theta^*\mid\theta^{(t)})$, accept with
probability

$$
\alpha = \min\left(1,\ \frac{f(x\mid\theta^*)p(\theta^*)\,q(\theta^{(t)}\mid\theta^*)}{f(x\mid\theta^{(t)})p(\theta^{(t)})\,q(\theta^*\mid\theta^{(t)})}\right).
$$

The posterior's unknown normalizing constant cancels in the ratio -- why MCMC avoids the
intractable integral. Detailed balance ($p(\theta)T(\theta\to\theta')=p(\theta')T(\theta'\to
\theta)$) guarantees $p(\theta\mid x)$ is stationary, so ergodic averages converge to posterior
expectations. For symmetric proposals, $\alpha$ reduces to the likelihood-times-prior ratio
(classical Metropolis).

**Gibbs sampling.** When each full conditional $p(\theta_j\mid\theta_{-j},x)$ is available in
closed form, cycle through drawing each $\theta_j$ from it directly. This is MH with the proposal
equal to the exact full conditional, giving acceptance probability identically $1$ (Exercise 2).
Attractive when conjugacy gives tractable conditionals, but can mix slowly under strong posterior
correlation.

## Convergence Diagnostics

- **Trace plots.** $\theta^{(t)}$ vs. $t$; well-mixed chains look like stationary noise, poor
  mixing shows slow drift or long excursions.
- **R-hat (Gelman-Rubin).** Run $m\geq2$ chains from dispersed starts; with within-chain variance
  $W$ and between-chain variance $B$,

$$
\hat R = \sqrt{\frac{\frac{n-1}{n}W+\frac1n B}{W}}.
$$

  $\hat R\approx1$ indicates convergence; $\hat R\gg1$ means chains disagree.
- **Effective sample size (ESS).** MCMC draws are autocorrelated; with autocorrelation $\rho_k$,

$$
\mathrm{ESS} = \frac{n}{1+2\sum_{k=1}^\infty \rho_k}.
$$

  Low ESS relative to $n$ signals slow mixing; Monte Carlo SEs should use ESS, not $n$.

## Posterior Predictive Checks

Simulate replicated data from $p(x^{rep}\mid x) = \int f(x^{rep}\mid\theta)p(\theta\mid x)\,d\theta$
using posterior draws $\theta^{(1)},\dots,\theta^{(S)}$: draw $x^{rep,(s)}\sim f(\cdot\mid
\theta^{(s)})$ for each $s$. Compute a test statistic $T(x)$ (max, skewness, number of zeros) on
observed and replicated data; if $T(x)$ falls in the tail of the replicated distribution, the
model fails to capture that feature -- a Bayesian analogue of goodness-of-fit that propagates full
posterior uncertainty rather than conditioning on a point estimate.

## Worked Example

$15$ trials, $x=11$ successes, $\mathrm{Binomial}(15,p)$, prior $p\sim\mathrm{Beta}(2,2)$.

**Posterior.** $p\mid x\sim\mathrm{Beta}(13,6)$. Mean $13/19\approx0.684$ vs. MLE $11/15\approx
0.733$ -- the prior pulls slightly toward $0.5$. A 95% credible interval (2.5/97.5 percentiles of
$\mathrm{Beta}(13,6)$) is approximately $[0.454,0.870]$.

```python
from scipy.stats import beta
a_post, b_post = 2 + 11, 2 + 4
ci = beta.ppf([0.025, 0.975], a_post, b_post)
print(a_post / (a_post + b_post), ci)  # approx 0.684 [0.454 0.870]
```

**MCMC cross-check.** A random-walk Metropolis sampler recovers the same closed-form posterior:

```python
import numpy as np

def log_post(p, x=11, n=15, a=2, b=2):
    if p <= 0 or p >= 1:
        return -np.inf
    return x*np.log(p) + (n-x)*np.log(1-p) + (a-1)*np.log(p) + (b-1)*np.log(1-p)

rng = np.random.default_rng(0)
p_cur, draws = 0.5, []
for _ in range(20000):
    p_prop = p_cur + rng.normal(0, 0.08)
    if np.log(rng.uniform()) < log_post(p_prop) - log_post(p_cur):
        p_cur = p_prop
    draws.append(p_cur)
draws = np.array(draws[2000:])
print(draws.mean(), np.percentile(draws, [2.5, 97.5]))  # matches 0.684, [0.454, 0.870]
```

## Exercises

### Exercise 1

Derive the Normal-Normal posterior formulas by completing the square: $X_i\mid\mu\sim
N(\mu,\sigma^2)$ iid, $\mu\sim N(\mu_0,\tau_0^2)$, show $\mu\mid x\sim N(\mu_n,\tau_n^2)$ with
$1/\tau_n^2=1/\tau_0^2+n/\sigma^2$.

<details>
<summary>Solution</summary>

Unnormalized posterior: $\exp\left(-\frac1{2\sigma^2}\sum_i(x_i-\mu)^2\right)\exp\left(-\frac{(\mu-\mu_0)^2}{2\tau_0^2}\right)$.
Expand $\sum_i(x_i-\mu)^2 = n\mu^2-2n\bar x\mu+\text{const}$. Collecting the $\mu$-dependent
exponent terms:

$$
-\frac12\left[\left(\frac{n}{\sigma^2}+\frac1{\tau_0^2}\right)\mu^2 - 2\left(\frac{n\bar x}{\sigma^2}+\frac{\mu_0}{\tau_0^2}\right)\mu\right] = -\frac{A}{2}\mu^2 + B\mu,
$$

with $A=n/\sigma^2+1/\tau_0^2$, $B=n\bar x/\sigma^2+\mu_0/\tau_0^2$. Completing the square,
$-\frac{A}2\mu^2+B\mu = -\frac{A}2(\mu-B/A)^2 + \text{const}$, the kernel of $N(B/A,\,1/A)$.
Reading off: $\tau_n^2=1/A$, i.e. $1/\tau_n^2 = n/\sigma^2+1/\tau_0^2$ (precisions add), and
$\mu_n=B/A=\tau_n^2(n\bar x/\sigma^2+\mu_0/\tau_0^2)$, exactly the stated formulas.

</details>

### Exercise 2

Show Gibbs sampling is Metropolis-Hastings with acceptance probability identically $1$: for
$\theta=(\theta_1,\theta_2)$, take proposal $q(\theta_1^*\mid\theta_1^{(t)},\theta_2) = p(\theta_1^*
\mid\theta_2,x)$ and show the MH ratio simplifies to $1$.

<details>
<summary>Solution</summary>

MH ratio: $\alpha=\min\left(1,\frac{\pi(\theta^*)q(\theta_1^{(t)}\mid\theta_1^*,\theta_2)}
{\pi(\theta^{(t)})q(\theta_1^*\mid\theta_1^{(t)},\theta_2)}\right)$. Write $\pi(\theta_1,\theta_2) =
p(\theta_1\mid\theta_2,x)p(\theta_2\mid x)$. By construction $q(\theta_1^*\mid\theta_1^{(t)},
\theta_2)=p(\theta_1^*\mid\theta_2,x)$ (depends only on $\theta_2$, not $\theta_1^{(t)}$), and
symmetrically for the reverse proposal. Substituting:

$$
\frac{p(\theta_1^*\mid\theta_2,x)p(\theta_2\mid x)\cdot p(\theta_1^{(t)}\mid\theta_2,x)}{p(\theta_1^{(t)}\mid\theta_2,x)p(\theta_2\mid x)\cdot p(\theta_1^*\mid\theta_2,x)} = 1,
$$

every factor in the numerator exactly cancels a matching factor in the denominator. So $\alpha=1$:
every Gibbs proposal is accepted, confirming Gibbs is MH with proposal equal to the exact full
conditional, still targeting the correct joint stationary distribution via the same
detailed-balance argument.

</details>

### Exercise 3

For the worked example ($p\mid x\sim\mathrm{Beta}(13,6)$), compute $P(p>0.5\mid x)$ as an
incomplete beta function with a numeric value, and explain why this has no exact frequentist
confidence-interval analogue.

<details>
<summary>Solution</summary>

$$
P(p>0.5\mid x) = \int_{0.5}^1 \frac{p^{12}(1-p)^5}{B(13,6)}\,dp = 1 - I_{0.5}(13,6),
$$

the complement of the regularized incomplete beta function (the $\mathrm{Beta}(13,6)$ CDF at
$0.5$). Numerically, `1 - beta.cdf(0.5, 13, 6)` $\approx 0.923$: about a 92.3% posterior
probability that $p>0.5$.

**No frequentist analogue.** A frequentist procedure makes probability statements about itself
under repeated sampling for a *fixed* $p$; it cannot assign a probability to "$p>0.5$" because $p$
is a constant, not a random variable, so that event has probability $0$ or $1$ deterministically,
not something computed from data. Only in the Bayesian framework, where $p$ has a genuine
posterior distribution, does $P(p>0.5\mid x)=0.923$ have a direct interpretation as a degree of
belief conditional on the data.

</details>

### Exercise 4

For a two-component Gaussian mixture with an exchangeable prior, explain via detailed balance why
a well-mixing MCMC chain is expected to visit both component labelings, why this makes the raw
posterior mean of $\mu_1$ a poor summary, and propose a fix.

<details>
<summary>Solution</summary>

**Why label switching occurs.** As in Lesson 2, $f(x;\theta)=f(x;\sigma(\theta))$ for the
label-swap permutation $\sigma$. With an exchangeable prior, $p(\theta\mid x)\propto
f(x;\theta)p(\theta) = f(x;\sigma(\theta))p(\sigma(\theta)) \propto p(\sigma(\theta)\mid x)$: the
posterior has (at least) two symmetric modes of equal height. A sampler correctly targeting this
posterior via detailed balance, if it mixes well enough to cross the low-density region between
modes, will visit both labelings over a long run, spending roughly equal time in each -- this is
correct behavior, not a malfunction; a chain that never switches is more likely evidence of poor
mixing.

**Why the raw mean of $\mu_1$ is poor.** If draws split roughly 50/50 between labelings, averaging
$\mu_1$ mixes draws from "the low-mean component" and "the high-mean component," producing a value
near the midpoint of the two true means -- which may correspond to neither subpopulation and can
sit in a region of near-zero mixture density.

**Fix.** Apply a post-hoc identifiability constraint to every draw before summarizing: e.g. at
each iteration, relabel so $\mu_1^{(s)}<\mu_2^{(s)}$ (swapping the corresponding $\pi$'s and
$\sigma^2$'s together), then compute the posterior mean of $\mu_1$ after relabeling. More
sophisticated loss-based relabeling algorithms exist for cases where a simple ordering constraint
is ambiguous (e.g. overlapping components).

</details>
