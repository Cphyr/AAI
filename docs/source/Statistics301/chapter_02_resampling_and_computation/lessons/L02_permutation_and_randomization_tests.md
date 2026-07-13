# Permutation & Randomization Tests

## Motivation

The $t$-, $F$-, and chi-squared tests derive null distributions from parametric assumptions
that hold exactly only in idealized settings and approximately via CLT arguments in large
samples. These approximations can be unreliable for small samples, non-normal data, or
nonstandard statistics. Permutation tests instead build the null distribution directly from the
data, assuming only that under $H_0$ the group labels are **exchangeable** with the outcomes --
often far weaker than distributional normality, and, in randomized experiments, guaranteed by
the randomization mechanism itself (hence Fisher's term "randomization test").

## Exchangeability and the Exact Null Distribution

$Z_1,\dots,Z_n$ are **exchangeable** if $(Z_1,\dots,Z_n)\stackrel d=(Z_{\pi(1)},\dots,Z_{\pi(n)})$
for every permutation $\pi$. iid variables are exchangeable, but exchangeability is strictly
weaker (e.g. sampling without replacement from a finite population).

Consider a two-sample problem: $X_1,\dots,X_{n_1}$ (group A), $Y_1,\dots,Y_{n_2}$ (group B),
pooled into $N=n_1+n_2$ values with labels $G_1,\dots,G_N$. Under the **sharp null** (treatment
has zero effect on every unit -- each unit's outcome is identical under either label), labels
are independent of values, and the joint data distribution is invariant to which $n_1$ units are
labeled A. This licenses an exact test: enumerate all $\binom Nn_1$ label assignments, compute
$T$ (e.g. $\bar X-\bar Y$) for each, and use the resulting discrete distribution as the exact
null. The one-sided p-value is

$$
p = \frac{1}{\binom{N}{n_1}}\sum_{\pi} \mathbb 1\{T(\pi)\ge T_{obs}\}.
$$

This is **exact** for any finite $N$, with no asymptotic approximation and no assumption on $F$
beyond the sharp null and exchangeability -- the central appeal over asymptotic tests.

## Monte Carlo Approximation

$\binom Nn_1$ is typically infeasible to enumerate (e.g. $\binom{40}{20}\approx1.4\times10^{11}$).
Instead draw $B$ random permutations $\pi_1,\dots,\pi_B$ and estimate

$$
\hat p = \frac{1+\#\{b: T(\pi_b)\ge T_{obs}\}}{B+1}.
$$

Including the observed statistic as one of $B+1$ exchangeable draws guarantees $\hat p>0$ and
yields a p-value that is **exactly valid** (not just asymptotically) for any finite $B$:
rejecting when $\hat p\le\alpha$ controls Type I error at level $\alpha$ exactly. $B$ only
affects resolution/stability of $\hat p$, not validity; $B=9999$ is a common default for
$\alpha=0.05$.

## Choice of Test Statistic

Exactness holds for any $T$; **power** depends on matching $T$ to the alternative:

- Location shift: $\bar X-\bar Y$ or the two-sample $t$-statistic (the latter also standardizes
  across permutations for unequal variances).
- Scale/dispersion: variance ratio or Levene-type transformed differences.
- General distributional difference: Kolmogorov-Smirnov or Cramer-von Mises statistics.
- Correlation/regression: permute the pairing of $X_i,Y_i$ and use $\hat\rho$ or $\hat\beta$ to
  test independence.

$T$ should be fixed before inspecting which choice gives the smallest p-value; ex-post statistic
selection reintroduces multiple-testing bias even though each individual p-value is exact.

## Permutation vs. Bootstrap Testing

| | Permutation | Bootstrap |
|---|---|---|
| Resampling | Without replacement (relabel) | With replacement |
| Target | Exact null under exchangeability | Approximate sampling dist. of a statistic |
| Validity | Exact, finite $N$, under sharp null | Asymptotic in general |
| Requires | Exchangeability under $H_0$ | Consistency of $\hat F_n$ (or null model) for $F$ |

Bootstrapping under a null requires resampling from a null-restricted model (e.g. pool the
groups and resample from the pooled empirical distribution) or centering appropriately (test
$H_0:\mu_A=\mu_B$ via $\bar X^*-\bar Y^*-(\bar X-\bar Y)$ vs. $\bar X-\bar Y$). Naively checking
whether 0 lies in a difference-of-means bootstrap CI is a valid, asymptotically justified test,
but it targets a different hypothesis than the sharp null permutation tests exactly.

## Stratified and Restricted Permutations

- **Stratified/blocked**: if randomization was within blocks (e.g. per clinic), permute labels
  only within each block, preserving exchangeability conditional on block membership.
- **Paired/matched**: for pairs $(X_i,Y_i)$, the exchangeable unit is the within-pair sign, not
  the label vector -- randomly flip the sign of $D_i=X_i-Y_i$ independently ($2^n$ possible sign
  patterns), matching a randomized paired design.
- **Restricted designs** (Latin squares, constrained randomization): enumerate only the
  permutations actually reachable under the true randomization scheme; using unrestricted
  permutations when the design was restricted overstates null variability.

## Limitations of Permutation-Based Confidence Intervals

- CIs require **inverting** the test (find $\theta_0$ values for which $H_0:\theta=\theta_0$
  fails to reject), needing a full permutation test at every candidate $\theta_0$ -- expensive,
  typically via grid search.
- The sharp null (identical potential outcomes under either label) does not correspond to "no
  difference in means" under heterogeneous effects; a test can reject the sharp null even when
  the average effect is zero. Inversion-based CIs target the sharp null, not necessarily the ATE.
- Tractable inversion typically assumes a constant additive shift model
  ($Y_i(1)=Y_i(0)+\theta$), an extra assumption beyond what the test itself requires; if false,
  the resulting "CI" can be miscalibrated.
- Resolution is bounded by $1/(B+1)$ or $1/\binom{N}{n_1}$, giving coarser, less smooth
  boundaries than bootstrap or asymptotic CIs.

```python
import numpy as np
def permutation_test_mean_diff(x, y, B=9999, seed=0):
    rng = np.random.default_rng(seed)
    pooled, n1 = np.concatenate([x, y]), len(x)
    obs = x.mean() - y.mean()
    count = sum(1 for _ in range(B)
                if (rng.permutation(pooled)[:n1].mean()
                    - rng.permutation(pooled)[n1:].mean()) >= obs)
    return (1 + count) / (B + 1)
```

## Worked Example

$A=(12,15,18,20)$, $B=(22,25,28,30)$, $n_1=n_2=4$, $N=8$. Observed $\bar A-\bar B=-10.0$. There
are $\binom84=70$ label assignments. Because the two groups occupy disjoint, well-separated
ranges, full enumeration shows only the original assignment and its mirror ($\bar B-\bar
A=+10.0$) reach $|\bar A^*-\bar B^*|\ge10.0$, giving exact two-sided $p=2/70\approx0.0286$ --
computed with no distributional assumption despite $n=4$ per group. A $t$-test on the same data
gives $t\approx-4.06$ on 6 df, two-sided $p\approx0.0066$: noticeably smaller than the exact
combinatorial answer, though both reject at $\alpha=0.05$.

## Exercises

### Exercise 1

Show that the exact permutation distribution of $T=\bar X^*-\bar Y^*$ has mean 0, and derive its
variance in terms of the pooled sample variance $S_N^2$ of $Z_1,\dots,Z_N$.

<details>
<summary>Solution</summary>

Fix the pooled data with mean $\bar Z$ and variance $S_N^2$. Assigning $n_1$ of $N$ units to
group A uniformly at random is simple random sampling without replacement, so by classical
finite-population sampling theory, $E[\bar X^*]=\bar Z$ and
$\text{Var}(\bar X^*)=\frac{S_N^2}{n_1}\cdot\frac{n_2}{N}$. Since $\bar Y^*=\frac{N\bar
Z-n_1\bar X^*}{n_2}$ is deterministic given $\bar X^*$,

$$
T=\bar X^*-\bar Y^* = \frac{N}{n_2}(\bar X^*-\bar Z),
$$

so $E[T]=0$ regardless of the pooled data values, confirming validity under the sharp null.
Then

$$
\text{Var}(T)=\left(\frac{N}{n_2}\right)^2\text{Var}(\bar X^*) = \frac{NS_N^2}{n_1n_2} = S_N^2\left(\frac1{n_1}+\frac1{n_2}\right),
$$

matching the pooled-variance $t$-test's variance formula (using $S_N^2$ from pooled data rather
than a within-group estimate), explaining why the permutation and pooled $t$-test tend to agree
when $N$ is not too small.

</details>

### Exercise 2

For a paired design with $n=3$ and $D=(4,-1,6)$, enumerate the sign-flip null distribution of
$\bar D^*$ (all $2^3=8$ patterns) and compute the exact two-sided p-value for $T_{obs}=3.0$.

<details>
<summary>Solution</summary>

| Signs | Sum | $\bar D^*$ |
|---|---|---|
| $+,+,+$ | 9 | 3.00 |
| $+,+,-$ | -3 | -1.00 |
| $+,-,+$ | 11 | 3.67 |
| $+,-,-$ | -1 | -0.33 |
| $-,+,+$ | 1 | 0.33 |
| $-,+,-$ | -11 | -3.67 |
| $-,-,+$ | 3 | 1.00 |
| $-,-,-$ | -9 | -3.00 |

Patterns with $|\bar D^*|\ge3.00$: $3.00, 3.67, -3.67, -3.00$ -- 4 of 8, so $p=4/8=0.5$. With
$n=3$, the minimum achievable two-sided p-value is $2/8=0.25$, so no sign-flip test at this
sample size can reach $\alpha=0.05$: exactness does not imply adequate power.

</details>

### Exercise 3

Explain why selecting, from the same data, whichever of $m$ candidate test statistics gives the
smallest permutation p-value fails to control Type I error, and bound the error rate assuming
(worst case) independence of the $m$ p-values under $H_0$.

<details>
<summary>Solution</summary>

Each $p_k$ is marginally (super-)uniform under $H_0$, so $P(p_k\le\alpha)\le\alpha$ for a single
prespecified $k$. But reporting $p_{\min}=\min_k p_k$ and rejecting if $p_{\min}\le\alpha$ is a
different procedure, with Type I error $P(\min_k p_k\le\alpha\mid H_0)$, not the marginal rate
for one test -- exactness of each individual p-value says nothing about the distribution of
their minimum. Under independence,

$$
P(p_{\min}\le\alpha\mid H_0) = 1-(1-\alpha)^m,
$$

also bounded above by $m\alpha$ (Bonferroni). For $\alpha=0.05, m=5$:
$1-(0.95)^5\approx0.226$, over four times nominal. In practice the $m$ statistics are
positively correlated (same permuted data), which reduces but does not eliminate the inflation;
valid correction requires prespecifying $T$ or permuting a single combined (e.g. max) statistic
rather than taking the min of separately computed p-values.

</details>

### Exercise 4

A cluster-randomized trial assigns 3 of 6 classrooms (25 students each) to treatment. Explain
why permuting individual student labels is invalid, describe the correct scheme, and compare the
number of valid randomizations under each.

<details>
<summary>Solution</summary>

The physical randomization chose 3 of 6 classrooms uniformly (among $\binom63=20$ choices); all
students in a chosen classroom share its label. The exchangeable unit -- the unit the actual
randomization permuted -- is the classroom, not the student: two students in the same classroom
have perfectly dependent labels, not independently exchangeable ones. Permuting student labels
individually assumes each of $\binom{150}{75}$ student subsets is equally likely, a scheme never
actually used (it could split a classroom 60/40, impossible under the true design). This targets
the wrong null distribution and is typically anti-conservative, since it ignores intraclass
correlation from shared classroom-level factors, treating 150 correlated observations as 150
independent ones.

The correct scheme permutes classroom labels: choose 3 of 6 classrooms (all $\binom63=20$
possible assignments) as "treatment," assign every student in a chosen classroom accordingly,
and recompute $T$ for each of the 20 assignments. Unlike $\binom{150}{75}\approx3.9\times10^{43}$,
this is exactly enumerable, and more importantly is the *correct* target regardless of
computational feasibility. With only 20 possible values, the minimum achievable two-sided
p-value is $2/20=0.10$, correctly reflecting that the effective sample size for inference is 6
clusters, not 150 students.

</details>
