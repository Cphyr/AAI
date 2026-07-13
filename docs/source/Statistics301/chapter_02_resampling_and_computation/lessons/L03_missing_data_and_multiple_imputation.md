# Missing Data & Multiple Imputation

## Motivation

Nearly every real dataset has missing values, and the default response -- drop any row with a
missing entry ("complete-case analysis") -- is valid only under a strong, rarely-stated
assumption that is often false. This lesson develops Rubin's (1976) taxonomy of missing-data
mechanisms, shows precisely when complete-case analysis is biased, and develops principled
alternatives: likelihood-based and multiple-imputation approaches, including the pooling rules
needed for valid inference.

## MCAR, MAR, and MNAR

Let $Y=(Y_{obs},Y_{mis})$ and let $R$ be the response indicator ($R_j=1$ if $Y_j$ observed).
The mechanism is characterized by $P(R\mid Y_{obs},Y_{mis},\psi)$.

- **MCAR**: $P(R\mid Y_{obs},Y_{mis},\psi)=P(R\mid\psi)$ for all $Y_{obs},Y_{mis}$ -- missingness
  does not depend on any data value.
- **MAR**: $P(R\mid Y_{obs},Y_{mis},\psi)=P(R\mid Y_{obs},\psi)$ for all $Y_{mis}$ -- missingness
  may depend on *observed* data but, conditional on it, not on the unobserved values themselves.
- **MNAR**: neither holds; $P(R\mid Y)$ depends on $Y_{mis}$ even given $Y_{obs}$ (e.g. a
  patient drops out *because* their unrecorded blood pressure that day was high).

MCAR $\subset$ MAR $\subset$ MNAR. Critically, **MAR is untestable from observed data alone**:
since $Y_{mis}$ is never observed, no dataset can empirically distinguish MAR from MNAR.

## Why Complete-Case Analysis Is Biased Under MAR

To estimate $\mu=E[Y]$, complete-case (CC) analysis computes $\hat\mu_{CC}=E[Y\mid R=1]$. Under
MCAR, $R\perp Y$ so $E[Y\mid R=1]=\mu$: unbiased (though less precise, since $n_{obs}<n$).

With a single variable, MAR reduces to MCAR (no other observed variable to condition on), so CC
bias appears whenever missingness depends on $Y$ itself. The interesting case has an auxiliary
covariate $X$: suppose missingness in $Y$ depends on $X$ but, given $X$, not further on $Y$
(MAR). Then

$$
E[Y\mid R=1] = E_X\big[E[Y\mid X,R=1]\mid R=1\big] = E_X\big[E[Y\mid X]\mid R=1\big],
$$

using MAR to drop conditioning on $R$ inside. But $E_X[\cdot\mid R=1]$ averages over $X$'s
distribution **among complete cases**, which differs from the marginal $X$-distribution whenever
$R$ depends on $X$. If $E[Y\mid X]$ varies with $X$, then generally
$E_X[E[Y|X]\mid R=1]\ne E[Y]=\mu$: **CC is biased under MAR** whenever the variable driving
missingness is associated with the outcome. Bias vanishes only if $Y\perp X$ or $R\perp X$
(MCAR). The same logic extends to regression: CC is unbiased if missingness depends only on
already-included predictors, but generally biased if it depends on $Y$ itself.

## Likelihood-Based and EM Approaches

Under MAR plus ignorability, the observed-data likelihood
$L(\theta\mid Y_{obs})=\int f(Y_{obs},Y_{mis}\mid\theta)\,dY_{mis}$ can be maximized directly
without modeling $R$ at all -- **ignorable likelihood inference**. When the integral is
intractable, the **EM algorithm** (Dempster, Laird, Rubin, 1977) alternates:

- **E-step**: $Q(\theta\mid\theta^{(t)}) = E_{Y_{mis}\mid Y_{obs},\theta^{(t)}}[\log
  f(Y_{obs},Y_{mis}\mid\theta)]$.
- **M-step**: $\theta^{(t+1)}=\arg\max_\theta Q(\theta\mid\theta^{(t)})$.

Each iteration does not decrease the observed-data log-likelihood, and under regularity
converges to a stationary point. EM gives point estimates directly, but SEs require extra work
(Louis's method, bootstrap, Supplemented EM) -- a practical reason multiple imputation is often
preferred: valid SEs fall out of the procedure automatically.

## Multiple Imputation

**MI** (Rubin, 1987) replaces missing values with $M>1$ plausible draws from a model for
$Y_{mis}\mid Y_{obs}$, giving $M$ completed datasets. Each is analyzed with the standard
complete-data method, yielding $M$ estimates $\hat\theta_m$ and variances $\hat V_m$, then
pooled via Rubin's rules. Steps: (1) **Imputation** -- draw $Y_{mis}^{(m)}\sim P(Y_{mis}\mid
Y_{obs})$ independently for $m=1,\dots,M$; (2) **Analysis** -- fit the model to each $Y^{(m)}$;
(3) **Pooling** -- combine via Rubin's rules. MI beats single imputation because single
imputation treats guesses as truly observed, understating uncertainty.

## Rubin's Rules

$$
\bar\theta=\frac1M\sum_m\hat\theta_m, \qquad
\bar V_W=\frac1M\sum_m\hat V_m, \qquad
V_B=\frac1{M-1}\sum_m(\hat\theta_m-\bar\theta)^2,
$$

$$
V_{total} = \bar V_W + \left(1+\frac1M\right)V_B.
$$

The $V_B/M$ term corrects for finite $M$ (vanishes as $M\to\infty$). Inference uses
$t_\nu$ with

$$
\nu = (M-1)\left[1+\frac{\bar V_W}{(1+1/M)V_B}\right]^2,
$$

so $\hat\theta\pm t_{\nu,1-\alpha/2}\sqrt{V_{total}}$. The **fraction of missing information**,
$\widehat{FMI}=\frac{(1+1/M)V_B}{V_{total}}$, diagnoses how much missingness degrades precision.

## MICE

With multiple variables missing in an arbitrary pattern, a single joint model is often
infeasible (mixed continuous/binary/categorical types). **MICE** ("chained equations") iterates
univariate conditional models: for each variable $Y_j$ with missingness, fit $Y_j$ on all other
variables $Y_{-j}$ (linear/logistic/multinomial as appropriate) using current values (observed
or currently imputed) of $Y_{-j}$, draw imputed values from this fit, and cycle through all
variables repeatedly (10-20 iterations) until stable. Repeat the whole chain $M$ times
independently. MICE's conditional models are not guaranteed compatible with any single joint
density, but it performs well in practice and dominates applied work due to flexibility with
mixed types.

## Sensitivity Analysis for MNAR

Since MAR is untestable, suspected-MNAR analyses should include sensitivity analysis rather than
a single MAR-based imputation:

- **Pattern-mixture models**: model $Y$'s distribution separately by missingness pattern, with a
  specified (not estimated) sensitivity parameter $\delta$ shifting the imputed mean for missing
  cases; vary $\delta$ and report a "tipping point."
- **Selection models**: jointly model the outcome and $P(R\mid Y,X)$ explicitly (e.g. Heckman-type),
  with dependence on $Y_{mis}$ a weakly-identified parameter needing strong assumptions or an
  instrument.
- **Delta-adjustment MI**: shift MAR-based imputed values by a range of $\delta$, re-pool, and
  report how conclusions change across the plausible range rather than a single point estimate.

Reporting should always make the untestable assumption explicit and bound results across
plausible departures from MAR rather than presenting a MAR-based answer as assumption-free.

```python
import numpy as np
def pool_rubin(theta_hats, var_hats):
    M = len(theta_hats)
    theta_bar, V_W = np.mean(theta_hats), np.mean(var_hats)
    V_B = np.var(theta_hats, ddof=1)
    V_total = V_W + (1 + 1/M) * V_B
    nu = (M - 1) * (1 + V_W / ((1 + 1/M) * V_B)) ** 2
    fmi = (1 + 1/M) * V_B / V_total
    return theta_bar, V_total, nu, fmi
```

## Worked Example

Income $Y$ is 30% missing, depending on age $X$ but, given $X$, not further on $Y$ (MAR). With
$M=5$ MICE datasets, a regression of $Y$ on $X$ gives slopes $\hat\beta=(152,148,160,145,155)$,
variances $\hat V=(100,110,95,105,90)$.

$\bar\beta=760/5=152.0$. $\bar V_W=500/5=100.0$. Deviations $0,-4,8,-7,3$; squares
$0,16,64,49,9$, sum $138$; $V_B=138/4=34.5$. $V_{total}=100.0+1.2(34.5)=141.4$, SE
$=\sqrt{141.4}\approx11.89$. $\nu=4[1+100.0/41.4]^2=4(3.415)^2\approx46.66$.
$\widehat{FMI}=41.4/141.4\approx0.293$ (29% of variance from missingness). Using
$t_{46.66,0.975}\approx2.01$: CI $=152.0\pm2.01(11.89)\approx(128.1,175.9)$ -- notably wider than
the naive $152.0\pm1.96(10.0)=(132.4,171.6)$ a single-imputation analysis would give by ignoring
$V_B$.

## Exercises

### Exercise 1

Let $Y_i\stackrel{iid}\sim N(\mu,\sigma^2)$, $\sigma^2$ known, with $P(R_i=0\mid Y_i) =
\Phi(\gamma_0+\gamma_1 Y_i)$ (MNAR). To first order in small $\gamma_1$, derive the asymptotic
bias of $\hat\mu_{CC}=E[Y\mid R=1]$ and give its sign for $\gamma_1>0$.

<details>
<summary>Solution</summary>

Let $g(y)=1-\Phi(\gamma_0+\gamma_1 y) = P(R=1\mid Y=y)$. We want
$E[Y\mid R=1]=E[Yg(Y)]/E[g(Y)]$. Taylor-expand: $g(y)\approx g_0-\gamma_1 y\phi(\gamma_0)$,
$g_0=1-\Phi(\gamma_0)$. Then $E[Yg(Y)]\approx g_0\mu-\gamma_1\phi(\gamma_0)(\sigma^2+\mu^2)$ and
$E[g(Y)]\approx g_0-\gamma_1\phi(\gamma_0)\mu$. Using $1/(1-x)\approx1+x$,

$$
E[Y\mid R=1] \approx \mu - \frac{\gamma_1\phi(\gamma_0)\sigma^2}{g_0} + O(\gamma_1^2),
$$

after dropping the $O(\gamma_1^2)$ cross term. So

$$
\text{Bias}(\hat\mu_{CC}) \approx -\frac{\gamma_1\phi(\gamma_0)\sigma^2}{1-\Phi(\gamma_0)}.
$$

Since $\phi(\gamma_0)>0$ and $1-\Phi(\gamma_0)\in(0,1)$, the bias has the *opposite* sign of
$\gamma_1$: for $\gamma_1>0$ (larger $Y$ more likely missing), $\hat\mu_{CC}$ is biased
**downward**, since high values are underrepresented among complete cases. The magnitude scales
with $\sigma^2$ and $\gamma_1$, vanishing when $\gamma_1=0$ (MCAR).

</details>

### Exercise 2

Give the law-of-total-variance heuristic motivating Rubin's formula
$V_{total}=\bar V_W+(1+1/M)V_B$ as $M\to\infty$, identifying which term corresponds to which
piece of the decomposition.

<details>
<summary>Solution</summary>

View $\hat\theta_m=\theta^*(Y_{mis}^{(m)})$ as induced by the random draw $Y_{mis}^{(m)}\sim
P(Y_{mis}\mid Y_{obs})$, with $\hat V_m$ the corresponding complete-data sampling-variance
estimator. Decompose total uncertainty about $\theta$ as

$$
\text{Var}(\hat\theta_{true}) \approx \underbrace{E[\hat V_m]}_{\to\ \bar V_W} + \underbrace{\text{Var}_{Y_{mis}}(\hat\theta_m)}_{\to\ V_B},
$$

the first term being the average "within" complete-data sampling variance (had $Y_{mis}$ been
known) and the second the "between" variance due to not observing $Y_{mis}$. As $M\to\infty$,
by LLN, $\bar V_W\to E[\hat V_m]$ and $V_B\to\text{Var}_{Y_{mis}}(\hat\theta_m)$. The extra
$V_B/M$ accounts for $\bar\theta$ itself being an average of only $M$ (not infinite) draws, with
its own Monte Carlo variance $V_B/M$ around the $M=\infty$ limit -- ensuring $V_{total}$
reflects total uncertainty even for finite $M$.

</details>

### Exercise 3

MI imputes missing $Y$ via $\hat Y=\hat\beta_0+\hat\beta_1X+\hat\epsilon$,
$\hat\epsilon\sim N(0,\hat\sigma^2)$, but reuses the *same* fitted $\hat\beta_0,\hat\beta_1,
\hat\sigma^2$ across all $M$ imputations (only re-drawing $\hat\epsilon$). Explain why this
"improper" scheme still understates pooled SEs, and which term is affected.

<details>
<summary>Solution</summary>

Proper MI requires imputations to reflect both residual noise *and* parameter uncertainty in
$\hat\beta_0,\hat\beta_1,\hat\sigma^2$ (estimated from a finite sample, not the truth); a proper
scheme redraws parameters per imputation (e.g. from their posterior, or via bootstrapping the
complete cases before fitting). Reusing fixed point estimates across all $m$ means the *only*
source of cross-imputation variability is the residual noise draw, entirely omitting parameter
uncertainty.

Consequently $V_B=\frac1{M-1}\sum(\hat\theta_m-\bar\theta)^2$ understates the true
between-imputation variance -- any variation in $\hat\theta_m$ comes only from noise averaging,
not genuine uncertainty about $\beta$. Since $V_{total}=\bar V_W+(1+1/M)V_B$, an artificially
small $V_B$ directly yields artificially narrow pooled SEs and CIs -- a subtler version of the
standard single-imputation failure to propagate uncertainty, despite using the correct
regression functional form.

</details>

### Exercise 4

With MAR-based MI giving $\bar\theta_0=50.0$, and delta-adjustment showing
$\bar\theta(\delta)=50.0+0.32\delta$ (offsetting imputed values only), find the tipping point
$\delta^*$ at which $\bar\theta(\delta)$ crosses the clinical threshold of 45, and interpret it.

<details>
<summary>Solution</summary>

Set $50.0+0.32\delta=45.0 \Rightarrow \delta = -5.0/0.32 \approx -15.6$.

So if non-responders' true values average about 15.6 units below what the MAR model predicts
for them (given observed covariates), the estimate reaches exactly the threshold; any larger
downward departure reverses the conclusion. This translates an abstract MNAR worry into a
concrete, interpretable quantity: domain experts can judge whether a shift of that magnitude is
plausible given what is known about why high-value subjects might drop out. If plausible dropout
effects are smaller (e.g. 5-8 units), the conclusion is robust; if comparable or larger,
it should be reported as sensitive to the untestable MAR assumption. Tipping-point analysis does
not resolve untestability but reframes it as a calibrated, answerable question.

</details>
