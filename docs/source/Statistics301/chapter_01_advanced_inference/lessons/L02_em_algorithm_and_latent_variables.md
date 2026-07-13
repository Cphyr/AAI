# The EM Algorithm & Latent Variables

## Motivation

Lesson 1 assumed the log-likelihood is easy to differentiate. Mixture models, missing data,
hidden Markov models, and factor-analysis-type models break that assumption: their likelihoods
involve a sum or integral over unobserved latent variables, so the log cannot be pushed inside the
sum. The Expectation-Maximization (EM) algorithm sidesteps this by alternating between "filling
in" the latent variables using their current best guess and re-maximizing a simpler *complete-data*
likelihood. It underlies Gaussian mixture models and hidden Markov models, and gives a concrete,
algorithmic illustration of the Jensen's-inequality / KL-divergence ideas that reappear in the
variational and Bayesian treatments later in this course.

## Incomplete-Data Likelihood

Suppose we observe $X$ but the generative model also involves latent $Z$, with joint density
$f(x,z;\theta)$. The observed-data log-likelihood marginalizes out $Z$:

$$
\ell(\theta;x) = \log f(x;\theta) = \log \sum_z f(x,z;\theta).
$$

The sum inside the log is what makes direct maximization hard: it does not decompose into simple
per-observation terms. If $Z$ *were* observed, the complete-data log-likelihood $\log
f(x,z;\theta)$ would typically be easy to maximize in closed form. EM exploits this by working
with the complete-data likelihood in expectation over $Z$ given the current parameters.

## Deriving EM via Jensen's Inequality and the ELBO

For arbitrary $q(z)$,

$$
\ell(\theta;x) = \log\sum_z q(z)\frac{f(x,z;\theta)}{q(z)} \geq \sum_z q(z)\log\frac{f(x,z;\theta)}{q(z)} \equiv \mathcal L(q,\theta),
$$

by Jensen's inequality (log is concave), where $\mathcal L(q,\theta)$ is the **evidence lower
bound (ELBO)**. The gap is a KL divergence:

$$
\ell(\theta;x) - \mathcal L(q,\theta) = D_{KL}\big(q \,\|\, p(\cdot\mid x;\theta)\big) \geq 0,
$$

with $p(z\mid x;\theta)=f(x,z;\theta)/f(x;\theta)$ the posterior of $Z$. The bound is tight exactly
when $q(z)=p(z\mid x;\theta)$. EM performs coordinate ascent on $\mathcal L(q,\theta)$:

- **E-step.** Fix $\theta=\theta^{(t)}$, set $q(z)=p(z\mid x;\theta^{(t)})$, making the bound
  tight.
- **M-step.** Fix $q$, maximize $\mathcal L(q,\theta)$ over $\theta$. Since $\sum_z
  q(z)\log q(z)$ does not depend on $\theta$, this reduces to

$$
\theta^{(t+1)} = \arg\max_\theta Q(\theta\mid\theta^{(t)}), \qquad Q(\theta\mid\theta^{(t)}) = E_{Z\mid X;\theta^{(t)}}[\log f(X,Z;\theta)].
$$

## Monotonicity

**Claim.** $\ell(\theta^{(t+1)};x) \geq \ell(\theta^{(t)};x)$ for every EM iteration.

**Proof sketch.**

$$
\ell(\theta^{(t+1)};x) \geq \mathcal L(q^{(t)},\theta^{(t+1)}) \geq \mathcal L(q^{(t)},\theta^{(t)}) = \ell(\theta^{(t)};x).
$$

The first inequality is the ELBO bound at $\theta^{(t+1)}$ with $q^{(t)}$ still fixed (holds for
any $q$); the middle holds because the M-step chose $\theta^{(t+1)}$ to maximize $\mathcal
L(q^{(t)},\theta)$; the equality is E-step tightness. So the observed-data log-likelihood sequence
is non-decreasing, and (being bounded above in well-behaved models) converges. This does **not**
imply $\theta^{(t)}$ converges to a global -- or even local -- maximizer; see caveats below.

## Worked Example: Two-Component Gaussian Mixture

$Z_i\in\{1,2\}$, $P(Z_i=1)=\pi$, $X_i\mid Z_i=k \sim N(\mu_k,\sigma_k^2)$.

**E-step.** Responsibility of component $k$ for observation $i$:

$$
\gamma_{ik} = \frac{\pi_k^{(t)} N(x_i;\mu_k^{(t)},\sigma_k^{2(t)})}{\sum_j \pi_j^{(t)} N(x_i;\mu_j^{(t)},\sigma_j^{2(t)})}.
$$

**M-step.** Weighted MLE updates:

$$
\pi_k^{(t+1)} = \frac1n\sum_i\gamma_{ik}, \quad \mu_k^{(t+1)}=\frac{\sum_i\gamma_{ik}x_i}{\sum_i\gamma_{ik}}, \quad \sigma_k^{2(t+1)} = \frac{\sum_i\gamma_{ik}(x_i-\mu_k^{(t+1)})^2}{\sum_i\gamma_{ik}}.
$$

**Numeric snapshot.** With $x=(1.0,1.5,1.2,8.0,8.5,7.8)$, initializing $\mu_1^{(0)}=0,
\mu_2^{(0)}=9,\sigma_1^{2(0)}=\sigma_2^{2(0)}=1,\pi^{(0)}=0.5$, the E-step quickly assigns
$\gamma_{i1}\approx1$ to the first three points and $\gamma_{i2}\approx1$ to the last three, and
the M-step converges to $\hat\mu_1\approx1.23$, $\hat\mu_2\approx8.1$, $\hat\pi\approx0.5$.

```python
import numpy as np
from scipy.stats import norm

x = np.array([1.0, 1.5, 1.2, 8.0, 8.5, 7.8])
pi, mu1, mu2, s1, s2 = 0.5, 0.0, 9.0, 1.0, 1.0
for _ in range(20):
    w1 = pi * norm.pdf(x, mu1, np.sqrt(s1))
    w2 = (1 - pi) * norm.pdf(x, mu2, np.sqrt(s2))
    gamma = w1 / (w1 + w2)
    pi = gamma.mean()
    mu1 = (gamma * x).sum() / gamma.sum()
    mu2 = ((1 - gamma) * x).sum() / (1 - gamma).sum()
    s1 = (gamma * (x - mu1)**2).sum() / gamma.sum()
    s2 = ((1 - gamma) * (x - mu2)**2).sum() / (1 - gamma).sum()
print(pi, mu1, mu2, s1, s2)
```

## Convergence Caveats

- **Local optima.** EM is coordinate ascent on a non-convex surface, so it can converge to a local
  (not global) maximum. Standard fix: multi-start from many random initializations, keep the best
  final log-likelihood.
- **Label switching.** The mixture likelihood is invariant to permuting component labels, so
  labels are not identifiable: you must align components (e.g. sort by $\mu_k$) before averaging
  or comparing parameters across runs. This resurfaces as a central issue for the Bayesian/MCMC
  treatment of mixtures in Lesson 3.
- **Degenerate solutions.** With unrestricted per-component variances, the likelihood is
  unbounded: collapsing $\mu_k$ onto a single point with $\sigma_k^2\to0$ sends it to infinity.
  Mitigated by variance floors or a weakly informative prior.
- **Slow convergence.** EM's convergence rate is linear (not quadratic like Newton's method), and
  can be very slow when components overlap or responsibilities hover near $0.5$.

## Exercises

### Exercise 1

Derive the E-step and M-step for EM applied to a mixture of $K$ Poissons: $Z_i\in\{1,\dots,K\}$,
$P(Z_i=k)=\pi_k$, $X_i\mid Z_i=k\sim\mathrm{Poisson}(\lambda_k)$.

<details>
<summary>Solution</summary>

Complete-data log-likelihood: $\log f(x_i,k;\theta) = \log\pi_k + x_i\log\lambda_k -\lambda_k -
\log(x_i!)$.

**E-step.**

$$
\gamma_{ik} = \frac{\pi_k^{(t)}(\lambda_k^{(t)})^{x_i}e^{-\lambda_k^{(t)}}}{\sum_j \pi_j^{(t)}(\lambda_j^{(t)})^{x_i}e^{-\lambda_j^{(t)}}}
$$

($x_i!$ cancels between numerator and denominator).

**M-step.** $Q(\theta\mid\theta^{(t)}) = \sum_{i,k}\gamma_{ik}[\log\pi_k + x_i\log\lambda_k -
\lambda_k] + \text{const}$. For $\pi_k$, maximize $\sum_{i,k}\gamma_{ik}\log\pi_k$ subject to
$\sum_k\pi_k=1$; a Lagrange multiplier gives $\pi_k \propto \sum_i\gamma_{ik}$, so
$\pi_k^{(t+1)}=\frac1n\sum_i\gamma_{ik}$ after normalizing. For $\lambda_k$, differentiate:
$\sum_i\gamma_{ik}(x_i/\lambda_k - 1) = 0$, giving

$$
\lambda_k^{(t+1)} = \frac{\sum_i\gamma_{ik}x_i}{\sum_i\gamma_{ik}},
$$

a responsibility-weighted mean of counts, directly analogous to the Gaussian mixture's $\mu_k$
update.

</details>

### Exercise 2

Prove the KL decomposition $\log f(x;\theta) = \mathcal L(q,\theta) + D_{KL}(q\|p(\cdot\mid
x;\theta))$, then use it to give a non-Jensen proof that $\mathcal L(q,\theta)\leq\log
f(x;\theta)$ with equality iff $q=p(\cdot\mid x;\theta)$.

<details>
<summary>Solution</summary>

Expand the right side: $\mathcal L(q,\theta) + D_{KL}(q\|p) = \sum_z q(z)\log\frac{f(x,z;\theta)}{q(z)}
+ \sum_z q(z)\log\frac{q(z)}{p(z\mid x;\theta)} = \sum_z q(z)\log\frac{f(x,z;\theta)}{p(z\mid x;\theta)}$.
Substituting $p(z\mid x;\theta)=f(x,z;\theta)/f(x;\theta)$, the ratio inside equals $f(x;\theta)$
for every $z$, so the sum is $\log f(x;\theta)\sum_z q(z) = \log f(x;\theta)$, proving the
identity.

**Non-Jensen bound proof.** $D_{KL}(q\|p)\geq0$ always, with equality iff $q=p$ a.e. (a standard
fact about relative entropy). Rearranging the identity, $\mathcal L(q,\theta) = \log f(x;\theta) -
D_{KL}(q\|p) \leq \log f(x;\theta)$, with equality iff $q(z)=p(z\mid x;\theta)$ for all $z$ with
$q(z)>0$. This recovers the ELBO inequality without Jensen and makes explicit that the E-step
closes exactly a KL gap.

</details>

### Exercise 3

$m=10$ coin tosses; each toss independently used coin $A$ (bias $\theta$, unknown) with
probability $\pi=0.5$ or coin $B$ (bias $0.5$, known) with probability $0.5$; coin identity per
toss is latent. Observed: $7$ heads. Derive the M-step update for $\theta$ in terms of E-step
responsibilities, and explain why EM here is not just a one-step closed-form computation.

<details>
<summary>Solution</summary>

**E-step.** Responsibility that a head (resp. tail) came from coin $A$:

$$
\gamma_H = \frac{0.5\,\theta^{(t)}}{0.5\,\theta^{(t)}+0.5\cdot0.5}, \qquad
\gamma_T = \frac{0.5(1-\theta^{(t)})}{0.5(1-\theta^{(t)})+0.5\cdot0.5}.
$$

**M-step.** Maximizing $\sum_j\gamma_j[y_j\log\theta+(1-y_j)\log(1-\theta)]$ over $\theta$ gives
the weighted-MLE update $\theta^{(t+1)} = \frac{7\gamma_H}{7\gamma_H+3\gamma_T}$.

**Why iterative, not one-shot.** $\gamma_H,\gamma_T$ depend on $\theta^{(t)}$ through the E-step
formula, so $\theta^{(t+1)}$ is a nonlinear function of $\theta^{(t)}$: this is a genuine
fixed-point iteration, not a closed-form MLE, precisely because the coin identity per toss is
never resolved. Start from $\theta^{(0)}=0.5$, alternate E- and M-steps until $\theta^{(t)}$
stabilizes.

</details>

### Exercise 4

Explain, using the Gaussian mixture's label-switching symmetry, why convergence of the
log-likelihood sequence $\ell(\theta^{(t)})$ does not imply convergence of $\theta^{(t)}$ to a
single point. Why is this harmless for prediction but harmful for interpreting individual
component parameters?

<details>
<summary>Solution</summary>

**Why likelihood convergence does not pin down $\theta$.** The mixture density $f(x;\theta) =
\sum_k\pi_k N(x;\mu_k,\sigma_k^2)$ is invariant under any permutation of component labels, since
it is a sum whose order does not matter. So the likelihood surface has $K!$ symmetric copies of
every local maximum. The monotonicity proof only concerns the scalar sequence
$\ell(\theta^{(t)})$; it says nothing about which of these symmetric maximizers (or which basin)
the parameter sequence settles near, and different EM runs (or numerical noise near a near-tie)
can converge to label-permuted versions of the same solution.

**Harmless for prediction.** Any prediction depending only on $f(x;\theta)$ (density estimate,
posterior predictive probability, cluster assignment by max responsibility) is exactly invariant
to label permutation, so it is unaffected.

**Harmful for interpretation.** If "component 1" is meant to represent a specific
subpopulation and is tracked across bootstrap resamples or MCMC iterations (Lesson 3), an
arbitrary label permutation at each run destroys naive averaging of $\mu_1$ across runs -- you may
average "the low mean" from one run with "the high mean" from another. Fix: a post-hoc
identifiability constraint, e.g. sort components by $\hat\mu_k$ after fitting, applied uniformly
before any cross-run summary.

</details>
