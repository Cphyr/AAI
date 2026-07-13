# Mixed-Effects Models

## Motivation

Statistics 1 and 2 built regression and ANOVA on the assumption of independent, identically distributed errors. That assumption breaks down constantly in real data: repeated measurements on the same patient, students clustered in classrooms, classrooms clustered in schools, or longitudinal growth curves measured on the same subjects over time. Observations within a cluster tend to be correlated because they share unobserved cluster-level characteristics (a patient's baseline health, a school's resources). Ignoring this correlation and running ordinary least squares gives correct point estimates of population-average effects in simple cases but badly wrong standard errors, and it throws away information about how much clusters differ from one another.

Mixed-effects models (also called hierarchical or multilevel models) solve this by explicitly partitioning variability into a *fixed* part, shared population-level parameters we want to estimate, and a *random* part, cluster-specific deviations that are treated as draws from a distribution. This lesson develops the linear mixed model, its estimation theory (ML and REML), best linear unbiased prediction (BLUP) of random effects, and inference on variance components, including the notorious boundary problem in likelihood-ratio testing.

## The Linear Mixed Model

### Fixed vs. random effects

A factor's levels are treated as **fixed** when they are the specific levels of scientific interest and we want to estimate a separate, interpretable parameter for each (e.g., "treatment vs. placebo"). A factor's levels are treated as **random** when they are best thought of as a sample from a larger population of levels, and our interest is in the *variance* they induce and in generalizing beyond the observed levels (e.g., "these 40 schools represent all schools of this type"). A practical rule of thumb: if you would be willing to relabel or resample the levels and still ask the same scientific question, treat them as random.

### Random intercept/slope model in matrix form

For cluster $i = 1, \dots, m$ with $n_i$ observations, the linear mixed model is

$$
y_i = X_i \beta + Z_i b_i + \epsilon_i, \qquad b_i \sim N(0, D), \qquad \epsilon_i \sim N(0, \sigma^2 I_{n_i}),
$$

with $b_1, \dots, b_m$ mutually independent, independent across clusters of $\epsilon_i$, and $b_i \perp \epsilon_i$. Here $X_i$ is the $n_i \times p$ fixed-effects design matrix and $Z_i$ is the $n_i \times q$ random-effects design matrix.

For a **random intercept and slope** model with a single time covariate $t_{ij}$, $Z_i$ has columns $(1, t_{ij})$ and $b_i = (b_{0i}, b_{1i})^T \sim N(0, D)$ with

$$
D = \begin{pmatrix} \sigma_0^2 & \sigma_{01} \\ \sigma_{01} & \sigma_1^2 \end{pmatrix}.
$$

Marginalizing over $b_i$ gives $y_i \sim N(X_i\beta, V_i)$ with

$$
V_i = Z_i D Z_i^T + \sigma^2 I_{n_i}.
$$

Stacking clusters, $y = X\beta + Zb + \epsilon$ with $b \sim N(0, G)$, $G = I_m \otimes D$ block-diagonal, and marginal covariance $V = ZGZ^T + R$, $R = \sigma^2 I$.

### Nested vs. crossed effects

Random effects are **nested** when each level of one factor occurs within exactly one level of another (students within classrooms within schools). Nesting produces a block-diagonal (hierarchical) covariance structure, and $Z$ can be organized so that clusters at the coarsest level are conditionally independent. Random effects are **crossed** when the factors are not hierarchical, e.g., in a repeated-measures experiment where every subject rates every item, subject and item random effects are crossed: every subject-item combination is observed (or could be), and neither factor is contained within the other. Crossed designs require a $Z$ matrix that does not reduce to a per-cluster block structure, so the joint covariance mixes subject and item variance components across the whole dataset; software such as `lme4` handles this with sparse matrix methods rather than the closed-form cluster-wise algebra available for nested designs.

## Estimation: ML, REML, and BLUP

### ML vs. REML

Given variance components $\theta$ (the free parameters of $D$ and $\sigma^2$), maximum likelihood maximizes

$$
L(\beta, \theta) = \prod_{i=1}^m \phi\big(y_i; X_i\beta, V_i(\theta)\big)
$$

jointly over $\beta$ and $\theta$. As in ordinary regression, where the MLE of $\sigma^2$ divides by $n$ rather than $n-p$, the ML estimator of the variance components is downward biased because it does not account for the degrees of freedom used to estimate $\beta$.

**REML** (restricted/residual maximum likelihood) corrects this by maximizing the likelihood of error contrasts, linear combinations $K^Ty$ with $K^TX = 0$, that do not depend on $\beta$ at all:

$$
L_{REML}(\theta) \propto |V(\theta)|^{-1/2}\, |X^TV(\theta)^{-1}X|^{-1/2} \exp\!\left(-\tfrac12 (y-X\hat\beta(\theta))^T V(\theta)^{-1} (y-X\hat\beta(\theta))\right).
$$

REML variance-component estimates are unbiased in balanced designs and are the default for reporting variance components. However, because $L_{REML}$ depends on $X$ through the projection removing $\beta$, REML likelihoods are **not comparable across models with different fixed effects**; when comparing nested fixed-effects specifications by likelihood-ratio test, one must refit with ML.

### BLUP and shrinkage

For known $\theta$, the generalized least squares estimator is $\hat\beta = (X^TV^{-1}X)^{-1}X^TV^{-1}y$. The best linear unbiased predictor of the random effect is

$$
\hat b_i = D Z_i^T V_i^{-1}(y_i - X_i\hat\beta).
$$

This is a **shrinkage** estimator: it pulls the cluster's raw deviation toward zero (the population value), with the amount of shrinkage governed by cluster size and the ratio of between- to within-cluster variance. In the random-intercept-only case,

$$
\hat b_{0i} = \lambda_i (\bar y_i - x_i^T\hat\beta), \qquad \lambda_i = \frac{n_i \sigma_0^2}{n_i\sigma_0^2 + \sigma^2} \in (0,1).
$$

Small or noisy clusters (small $n_i$, or small $\sigma_0^2$ relative to $\sigma^2$) are shrunk more heavily toward the population mean; this is exactly the same shrinkage logic as in empirical Bayes estimation.

## Inference on Variance Components

### Likelihood-ratio testing for variance components (the boundary issue)

Testing $H_0: \sigma_0^2 = 0$ (no random intercept needed) is nonstandard: $0$ is on the **boundary** of the parameter space for a variance, so the usual regularity conditions for $-2\log\Lambda \to \chi^2_1$ under $H_0$ fail. Self and Liang (1987) showed that when testing a single variance component at the boundary, the asymptotic null distribution of the LRT statistic is a 50:50 mixture of a point mass at 0 and a $\chi^2_1$:

$$
-2\log\Lambda \;\xrightarrow{d}\; \tfrac12 \chi^2_0 + \tfrac12 \chi^2_1 \quad \text{(written } \bar\chi^2_{01}\text{)}.
$$

Practically, this means the naive p-value obtained by comparing the LRT statistic to a $\chi^2_1$ reference is roughly **twice too large** (conservative); the corrected p-value is approximately half the naive one. When testing $q \geq 2$ variance components jointly (e.g., a random slope's variance together with its covariance with the intercept), the reference distribution is a mixture of $\chi^2_{q-1}$ and $\chi^2_q$ with mixing weights that generally must be simulated or looked up.

### ICC

For the random-intercept model, the **intraclass correlation coefficient** is

$$
\rho = \frac{\sigma_0^2}{\sigma_0^2 + \sigma^2}.
$$

It equals both the proportion of total variance attributable to between-cluster differences, and the model-implied correlation between two observations $y_{ij}, y_{ik}$ ($j \ne k$) drawn from the same cluster $i$, since $Cov(y_{ij}, y_{ik}) = \sigma_0^2$ and $Var(y_{ij}) = \sigma_0^2+\sigma^2$ under the compound-symmetric marginal covariance implied by a single random intercept.

## Worked Example

Four schools (clusters), three students each, test scores:

| Group | Scores | Mean |
|---|---|---|
| 1 | 5, 7, 6 | 6.00 |
| 2 | 9, 10, 8 | 9.00 |
| 3 | 4, 3, 5 | 4.00 |
| 4 | 7, 8, 9 | 8.00 |

Grand mean $\bar y = 6.75$. This is a balanced one-way random-effects design, so the ANOVA-based variance component estimators coincide with REML.

$$
SSB = n\sum_i (\bar y_i - \bar y)^2 = 3\big[0.5625+5.0625+7.5625+1.5625\big] = 44.25, \quad MSB = 44.25/3 = 14.75
$$

$$
SSW = \sum_i\sum_j (y_{ij}-\bar y_i)^2 = 2+2+2+2 = 8, \quad MSW = 8/8 = 1
$$

So $\hat\sigma^2 = MSW = 1$ and, using $n=3$ observations per group,

$$
\hat\sigma_0^2 = \frac{MSB - MSW}{n} = \frac{14.75-1}{3} = 4.5833.
$$

$$
\widehat{ICC} = \frac{4.5833}{4.5833+1} = 0.821.
$$

Over 82% of the variance in scores is between-school; schools differ substantially.

**BLUP for group 3** (raw mean 4.00, furthest below the grand mean): shrinkage factor

$$
\lambda_3 = \frac{3(4.5833)}{3(4.5833)+1} = \frac{13.75}{14.75} = 0.932,
$$

$$
\hat b_{0,3} = \lambda_3(\bar y_3 - \hat\beta) = 0.932 \times (4.00 - 6.75) = -2.564.
$$

The predicted school-3 intercept is $6.75 - 2.564 = 4.186$, shrunk slightly toward the grand mean rather than sitting at the raw 4.00, because with only $n_i=3$ and $\sigma^2$ non-negligible relative to $\sigma_0^2$, some of the observed deviation is attributed to noise.

```python
import statsmodels.formula.api as smf
# model = smf.mixedlm("score ~ 1", data, groups=data["school"]).fit(reml=True)
# model.params, model.random_effects
```

## Exercises

### Exercise 1

Derive the marginal covariance matrix $V_i = Z_iDZ_i^T + \sigma^2 I_{n_i}$ for a random-intercept-only model ($Z_i = \mathbf{1}_{n_i}$, $D = \sigma_0^2$), and show explicitly that $V_i$ has the compound-symmetric form $\sigma^2 I_{n_i} + \sigma_0^2 J_{n_i}$ ($J$ the all-ones matrix). From this, derive the correlation between any two observations in the same cluster and confirm it equals the ICC formula given in the notes.

<details>
<summary>Solution</summary>

With $Z_i = \mathbf{1}_{n_i}$ (an $n_i \times 1$ column of ones) and $D = \sigma_0^2$ (a scalar), $Z_iDZ_i^T = \sigma_0^2 \mathbf{1}_{n_i}\mathbf{1}_{n_i}^T = \sigma_0^2 J_{n_i}$, where $J_{n_i}$ is the $n_i\times n_i$ matrix of all ones. Hence

$$
V_i = \sigma_0^2 J_{n_i} + \sigma^2 I_{n_i}.
$$

Element-wise, the diagonal entries are $\sigma_0^2 + \sigma^2$ (since $J_{ii}=1$ and $I_{ii}=1$) and the off-diagonal entries are $\sigma_0^2$ (since $J_{ij}=1, I_{ij}=0$ for $i\ne j$). This is exactly the compound-symmetric structure: constant variance, constant covariance.

The correlation between $y_{ij}$ and $y_{ik}$, $j\ne k$, in the same cluster is

$$
Corr(y_{ij},y_{ik}) = \frac{Cov(y_{ij},y_{ik})}{\sqrt{Var(y_{ij})Var(y_{ik})}} = \frac{\sigma_0^2}{\sigma_0^2+\sigma^2},
$$

since both variances equal $\sigma_0^2+\sigma^2$. This matches $\rho = \sigma_0^2/(\sigma_0^2+\sigma^2)$, confirming the ICC is precisely the within-cluster correlation implied by the random-intercept model, not merely a variance-decomposition heuristic.

</details>

### Exercise 2

Derive the BLUP $\hat b_i = DZ_i^TV_i^{-1}(y_i - X_i\beta)$ from the joint normality of $(y_i, b_i)$, and show it minimizes mean squared prediction error among linear unbiased predictors (equivalently, that it equals $E[b_i \mid y_i]$ under the normal model).

<details>
<summary>Solution</summary>

Under the model, $(y_i, b_i)$ is jointly Gaussian with

$$
\begin{pmatrix} y_i \\ b_i \end{pmatrix} \sim N\left( \begin{pmatrix} X_i\beta \\ 0 \end{pmatrix}, \begin{pmatrix} V_i & Z_iD \\ DZ_i^T & D \end{pmatrix} \right),
$$

because $Cov(y_i, b_i) = Cov(X_i\beta + Z_ib_i+\epsilon_i, b_i) = Z_i Cov(b_i,b_i) = Z_iD$, and $Var(b_i)=D$.

For a jointly Gaussian vector $(U,W)$ with $U\sim N(\mu_U,\Sigma_{UU})$, $W \sim N(\mu_W,\Sigma_{WW})$, and cross-covariance $\Sigma_{WU}$, the conditional expectation is

$$
E[W\mid U] = \mu_W + \Sigma_{WU}\Sigma_{UU}^{-1}(U-\mu_U).
$$

Applying this with $W = b_i$ ($\mu_W=0$), $U = y_i$ ($\mu_U = X_i\beta$), $\Sigma_{WU} = DZ_i^T$, $\Sigma_{UU}=V_i$:

$$
E[b_i \mid y_i] = DZ_i^T V_i^{-1}(y_i - X_i\beta) = \hat b_i.
$$

Optimality among linear unbiased predictors: any linear predictor $\tilde b_i = A y_i$ is unbiased for $b_i$ (in the sense $E[\tilde b_i - b_i]=0$ over the joint distribution) if $AX_i = 0$. Its MSE is $E\|Ay_i - b_i\|^2$. Writing $Ay_i - b_i = A(X_i\beta+Z_ib_i+\epsilon_i) - b_i = A\epsilon_i + (AZ_i-I)b_i$ (using $AX_i\beta=0$), and using independence of $\epsilon_i,b_i$,

$$
MSE(A) = \sigma^2 AA^T + (AZ_i-I)D(AZ_i-I)^T.
$$

Minimizing the trace of this quadratic form over $A$ subject to $AX_i=0$ is a constrained least-squares problem; setting the (constrained) gradient to zero yields exactly $A = DZ_i^TV_i^{-1}$ up to the projection removing the $X_i\beta$ component, reproducing $\hat b_i$. Since the conditional-expectation solution already satisfies $AX_i=0$ (because $Cov(b_i, X_i\beta)=0$ makes the predictor automatically orthogonal to the fixed part), the two derivations agree: the BLUP is simultaneously the conditional mean under normality and the minimum-MSE linear unbiased predictor, which is why it generalizes (as the "best linear unbiased predictor") to non-Gaussian settings via the Gauss-Markov-type argument alone.

</details>

### Exercise 3

For the balanced one-way random-effects model $y_{ij} = \mu + b_i + \epsilon_{ij}$, $b_i \sim N(0,\sigma_0^2)$, $\epsilon_{ij}\sim N(0,\sigma^2)$, $i=1,\dots,m$, $j=1,\dots,n$, derive $E[MSW]$ and $E[MSB]$ and show that the ANOVA-based moment estimators $\hat\sigma^2 = MSW$, $\hat\sigma_0^2 = (MSB-MSW)/n$ are unbiased for $\sigma^2,\sigma_0^2$.

<details>
<summary>Solution</summary>

Write $y_{ij} = \mu + b_i + \epsilon_{ij}$. Group mean $\bar y_i = \mu + b_i + \bar\epsilon_i$, grand mean $\bar y = \mu + \bar b + \bar{\bar\epsilon}$ where $\bar b = m^{-1}\sum_i b_i$, $\bar\epsilon_i = n^{-1}\sum_j \epsilon_{ij}$.

**Within groups:** $y_{ij}-\bar y_i = \epsilon_{ij}-\bar\epsilon_i$, which does not involve $b_i$ at all. So $SSW = \sum_i\sum_j(\epsilon_{ij}-\bar\epsilon_i)^2$ is exactly the usual within-group sum of squares of iid $N(0,\sigma^2)$ errors, with $m(n-1)$ degrees of freedom. Standard result (same as one-way fixed-effects ANOVA from Stats 2): $E[SSW] = m(n-1)\sigma^2$, so

$$
E[MSW] = E[SSW/(m(n-1))] = \sigma^2.
$$

**Between groups:** $\bar y_i - \bar y = (b_i - \bar b) + (\bar\epsilon_i - \bar{\bar\epsilon})$. Then

$$
SSB = n\sum_i(\bar y_i-\bar y)^2 = n\sum_i\big[(b_i-\bar b)+(\bar\epsilon_i-\bar{\bar\epsilon})\big]^2.
$$

Expand the square; cross terms vanish in expectation since $b_i \perp \epsilon$. So

$$
E[SSB] = n\, E\Big[\sum_i (b_i-\bar b)^2\Big] + n\, E\Big[\sum_i(\bar\epsilon_i - \bar{\bar\epsilon})^2\Big].
$$

For the first term, $b_1,\dots,b_m$ are iid $N(0,\sigma_0^2)$, so $\sum_i(b_i-\bar b)^2$ has expectation $(m-1)\sigma_0^2$ (standard unbiased-sample-variance identity). For the second term, $\bar\epsilon_i$ are iid $N(0,\sigma^2/n)$, so $\sum_i(\bar\epsilon_i-\bar{\bar\epsilon})^2$ has expectation $(m-1)\sigma^2/n$. Hence

$$
E[SSB] = n(m-1)\sigma_0^2 + (m-1)\sigma^2, \qquad E[MSB] = E[SSB/(m-1)] = n\sigma_0^2+\sigma^2.
$$

**Unbiasedness of the moment estimators:**

$$
E[\hat\sigma^2] = E[MSW] = \sigma^2, \qquad E[\hat\sigma_0^2] = \frac{E[MSB]-E[MSW]}{n} = \frac{(n\sigma_0^2+\sigma^2)-\sigma^2}{n} = \sigma_0^2.
$$

Both are unbiased. This ANOVA-based construction coincides with the REML estimators in the balanced case, which is why REML (not ML) is the natural generalization of the "$n-p$ correction" intuition from Stats 1/2 to variance components.

</details>

### Exercise 4

In a study, you fit a random-intercept model and a fixed-effects-only model by REML/ML and obtain a likelihood-ratio statistic $-2\log\Lambda = 3.5$ for $H_0:\sigma_0^2=0$ against $H_1:\sigma_0^2>0$ (one variance component, boundary case). (a) Compute the naive p-value using a $\chi^2_1$ reference distribution. (b) Compute the corrected p-value using the Self-Liang $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ mixture, and explain the direction of the discrepancy.

<details>
<summary>Solution</summary>

(a) Naive approach: treat $-2\log\Lambda = 3.5$ as $\chi^2_1$. We need $P(\chi^2_1 > 3.5)$. Since $\chi^2_1 = Z^2$ for $Z\sim N(0,1)$, $P(\chi^2_1>3.5) = P(|Z|>\sqrt{3.5}) = P(|Z|>1.8708) = 2(1-\Phi(1.8708))$. From the standard normal table, $\Phi(1.8708)\approx 0.9693$, so

$$
p_{naive} = 2(1-0.9693) = 2(0.0307) = 0.0614.
$$

(b) Under the correct mixture $\tfrac12\chi^2_0 + \tfrac12\chi^2_1$, the point mass $\chi^2_0$ contributes zero probability to the event $\{T>3.5\}$ for any $3.5>0$, so

$$
p_{corrected} = P(\bar\chi^2_{01} > 3.5) = \tfrac12 P(\chi^2_0>3.5) + \tfrac12 P(\chi^2_1>3.5) = 0 + \tfrac12(0.0614) = 0.0307.
$$

The corrected p-value is exactly half the naive one, as the general theory predicts: comparing the boundary LRT statistic to an ordinary $\chi^2_1$ reference is conservative (the naive p-value overstates the true p-value by a factor of 2), because half of the null sampling distribution's mass sits at exactly 0 (whenever the unconstrained REML/ML optimizer would want $\hat\sigma_0^2<0$, it is truncated to $0$, making $-2\log\Lambda=0$ on those samples) rather than following $\chi^2_1$ throughout. Using the naive $\chi^2_1$ p-value as a decision rule at $\alpha=0.05$ would fail to reject in this example ($0.0614>0.05$), while the corrected mixture test does reject ($0.0307<0.05$), so the boundary correction can change the conclusion of the test.

</details>
