# Survival Analysis

## Motivation

Time-to-event data, time until death, machine failure, customer churn, disease recurrence, has a feature that ordinary regression cannot handle: **censoring**. Many subjects have not yet experienced the event by the end of the study, so their exact event time is unknown, only a bound on it. Discarding censored subjects wastes information and biases estimates toward shorter observed times; treating censoring times as if they were event times is simply wrong. Survival analysis provides a coherent likelihood framework, and a set of nonparametric and semiparametric estimators, the Kaplan-Meier curve, the log-rank test, the Cox model, that correctly incorporate partial (censored) information.

## Survival, Hazard, and Cumulative Hazard Functions

### Censoring types

**Right censoring** is by far the most common: we know the event time exceeds the observed time (e.g., the subject is alive and event-free when the study ends, or is lost to follow-up). **Left censoring** occurs when we know the event happened *before* observation began but not exactly when (e.g., a screening test detects a condition already present, with unknown onset). **Interval censoring** occurs when we only know the event occurred within some interval $(L,R]$, common when subjects are examined only at periodic visits. Within right censoring, **Type I** censoring has a fixed censoring time (e.g., a study with a fixed end date), **Type II** censoring stops the study once a pre-specified number of events has occurred (common in reliability testing), and **random/independent censoring** allows censoring times to vary across subjects and be stochastic, provided they are independent of the (potential) event time given covariates, the key assumption that makes standard survival methods valid.

### Definitions and relations

Let $T\geq0$ be the (possibly unobserved) event time with density $f(t)$ and CDF $F(t)$. The **survival function** is

$$
S(t) = P(T>t) = 1-F(t).
$$

The **hazard function** is the instantaneous event rate given survival to $t$:

$$
h(t) = \lim_{\Delta t\to 0} \frac{P(t\leq T<t+\Delta t \mid T\geq t)}{\Delta t} = \frac{f(t)}{S(t)}.
$$

Since $f(t) = -S'(t)$, we have $h(t) = -S'(t)/S(t) = -\frac{d}{dt}\log S(t)$. Integrating,

$$
H(t) = \int_0^t h(u)\,du = -\log S(t) \quad\Longleftrightarrow\quad S(t) = \exp\big(-H(t)\big),
$$

where $H(t)$ is the **cumulative hazard function**. These identities let us move freely between $f$, $S$, $h$, and $H$: e.g. $f(t) = h(t)S(t) = h(t)\exp(-H(t))$.

## Kaplan-Meier Estimator and Greenwood's Formula

### Derivation of the KM estimator

Let $t_{(1)}<t_{(2)}<\dots<t_{(k)}$ be the distinct observed **event** times. At $t_{(j)}$, let $n_j$ be the number of subjects still at risk (neither having had the event nor been censored before $t_{(j)}$) and $d_j$ the number of events occurring exactly at $t_{(j)}$. Model the survival process via discrete hazards $\lambda_j = P(\text{event at } t_{(j)} \mid \text{at risk at } t_{(j)})$, one per distinct event time, and zero hazard elsewhere. Conditional on the risk sets, the number of events at $t_{(j)}$ given $n_j$ at risk is (treating individuals as exchangeable within the risk set) Binomial$(n_j,\lambda_j)$, so the nonparametric likelihood factors as

$$
L(\lambda_1,\dots,\lambda_k) = \prod_{j=1}^k \binom{n_j}{d_j}\lambda_j^{d_j}(1-\lambda_j)^{n_j-d_j}.
$$

This factors into separate terms for each $j$, so maximizing over each $\lambda_j$ independently gives the ordinary binomial MLE $\hat\lambda_j = d_j/n_j$. Since $S(t) = \prod_{j: t_{(j)}\leq t}(1-\lambda_j)$ under this discrete-hazard construction (the probability of surviving past $t$ is the probability of surviving each intervening discrete hazard), plugging in the MLEs gives the **Kaplan-Meier estimator**:

$$
\hat S(t) = \prod_{j:\, t_{(j)}\leq t} \left(1-\frac{d_j}{n_j}\right).
$$

Censored observations enter only through the risk sets $n_j$ (they reduce $n_j$ for all later event times) and contribute no $d_j$, which is exactly the correct partial-information treatment.

### Greenwood's formula

The variance of $\hat S(t)$ is obtained via the delta method applied to $\log\hat S(t) = \sum_{j:t_{(j)}\leq t}\log(1-d_j/n_j)$ (full derivation in Exercise 2):

$$
\widehat{Var}\big(\hat S(t)\big) = \hat S(t)^2 \sum_{j:\, t_{(j)}\leq t} \frac{d_j}{n_j(n_j-d_j)}.
$$

This is used to build confidence intervals for $S(t)$; because the normal-approximation interval $\hat S(t)\pm z_{\alpha/2}SE$ can extend outside $[0,1]$ (especially near the tails of follow-up where few subjects remain at risk), it is common practice to construct the interval on a transformed scale (e.g. $\log(-\log S(t))$) and back-transform.

## Log-Rank Test and Cox Proportional Hazards Model

### Log-rank test

To compare survival between two groups, at each distinct event time $t_{(j)}$ (pooling both groups) form a 2x2 table of group membership by event/no-event among those at risk, and compare observed events in group 1, $O_{1j}$, to their expectation under the null hypergeometric distribution (fixing the margins, $d_j$ total events and $n_{1j}, n_{2j}$ at risk in each group):

$$
E_{1j} = \frac{n_{1j}\,d_j}{n_j}, \qquad V_j = \frac{n_{1j}n_{2j}\,d_j\,(n_j-d_j)}{n_j^2(n_j-1)}.
$$

Summing over all event times and standardizing,

$$
Z = \frac{\sum_j (O_{1j}-E_{1j})}{\sqrt{\sum_j V_j}} \;\xrightarrow{d}\; N(0,1) \quad \text{under } H_0: S_1=S_2,
$$

equivalently $Z^2 \to \chi^2_1$. This is exactly a Mantel-Haenszel test stratified over the distinct event times.

### Cox proportional hazards model

The Cox model specifies the hazard for an individual with covariates $x$ as

$$
h(t\mid x) = h_0(t)\, \exp(x^T\beta),
$$

where $h_0(t)$ is an unspecified (nonparametric) baseline hazard, and the **proportional hazards (PH) assumption** is that the hazard ratio between any two covariate profiles, $\exp((x_1-x_2)^T\beta)$, is constant over time.

### Partial likelihood

Cox's key idea eliminates the infinite-dimensional nuisance $h_0(t)$ by conditioning, at each observed event time $t_{(j)}$, on the identity of the risk set $R(t_{(j)})$ and on the fact that exactly one event occurred, asking only *which* member of the risk set it was:

$$
P\big(\text{subject } (j) \text{ fails} \mid \text{one failure at } t_{(j)}, R(t_{(j)})\big) = \frac{h_0(t_{(j)})\exp(x_{(j)}^T\beta)}{\sum_{l\in R(t_{(j)})} h_0(t_{(j)})\exp(x_l^T\beta)} = \frac{\exp(x_{(j)}^T\beta)}{\sum_{l\in R(t_{(j)})}\exp(x_l^T\beta)},
$$

the baseline hazard cancels. Multiplying these conditional probabilities over all $k$ event times gives the **partial likelihood**

$$
L_p(\beta) = \prod_{j=1}^k \frac{\exp(x_{(j)}^T\beta)}{\sum_{l\in R(t_{(j)})}\exp(x_l^T\beta)}.
$$

Maximizing $\log L_p(\beta)$ by Newton-Raphson gives $\hat\beta$, and standard likelihood asymptotics (score, Wald, and likelihood-ratio tests, using the observed information from $\log L_p$) apply even though $L_p$ is not a full likelihood, a foundational (and nontrivial) result due to Cox (1972, 1975).

### Checking the PH assumption: Schoenfeld residuals

For covariate $k$ and event time $t_{(j)}$, the **Schoenfeld residual** compares the failing subject's observed covariate value to the risk-set-weighted expectation under the fitted model:

$$
r_{jk} = x_{(j)k} - \frac{\sum_{l\in R(t_{(j)})} x_{lk}\exp(x_l^T\hat\beta)}{\sum_{l\in R(t_{(j)})}\exp(x_l^T\hat\beta)}.
$$

Under a correctly specified PH model, these residuals should fluctuate randomly around zero with no systematic trend in time. A significant correlation between (scaled) Schoenfeld residuals and time, or a function of time, formally tested via the Grambsch-Therneau test, indicates that the effect of covariate $k$ is not actually constant over time (the PH assumption is violated), suggesting remedies such as stratification, time-varying coefficients, or splitting follow-up time.

## Worked Example

Two groups, $n=4$ each, times with event indicator (1=event, 0=censored):

Group A: 4(event), 6(event), 8(censored), 10(event). Group B: 3(event), 5(event), 7(event), 9(censored).

**Kaplan-Meier for the pooled sample** (8 subjects total):

| $t_{(j)}$ | at risk $n_j$ | events $d_j$ | factor $1-d_j/n_j$ | $\hat S(t_{(j)})$ |
|---|---|---|---|---|
| 3 | 8 | 1 | 7/8 | 0.8750 |
| 4 | 7 | 1 | 6/7 | 0.7500 |
| 5 | 6 | 1 | 5/6 | 0.6250 |
| 6 | 5 | 1 | 4/5 | 0.5000 |
| 7 | 4 | 1 | 3/4 | 0.3750 |
| 8 | 3 | 0 (censored) | -- | 0.3750 |
| 9 | 2 | 0 (censored) | -- | 0.3750 |
| 10 | 1 | 1 | 0/1 | 0.0000 |

**Greenwood variance at $t=7$** ($\hat S(7)=0.375$):

$$
\sum \frac{d_j}{n_j(n_j-d_j)} = \frac{1}{8\cdot7}+\frac{1}{7\cdot6}+\frac{1}{6\cdot5}+\frac{1}{5\cdot4}+\frac{1}{4\cdot3} = 0.01786+0.02381+0.03333+0.05+0.08333 = 0.20833
$$

$$
\widehat{Var}(\hat S(7)) = 0.375^2 \times 0.20833 = 0.14063\times0.20833 = 0.02930, \qquad SE = 0.1712.
$$

**Log-rank test** (Group A vs. Group B): tracking $n_{Aj}, n_{Bj}$ at each event time (A starts with 4 at risk, B starts with 4):

| $t_{(j)}$ | $n_{Aj}$ | $n_{Bj}$ | $n_j$ | $d_j$ | event in | $E_{Aj}=n_{Aj}d_j/n_j$ | $O_{Aj}-E_{Aj}$ | $V_j$ |
|---|---|---|---|---|---|---|---|---|
| 3 | 4 | 4 | 8 | 1 | B | 0.500 | -0.500 | 0.2500 |
| 4 | 4 | 3 | 7 | 1 | A | 0.571 | 0.429 | 0.2449 |
| 5 | 3 | 3 | 6 | 1 | B | 0.500 | -0.500 | 0.2500 |
| 6 | 3 | 2 | 5 | 1 | A | 0.600 | 0.400 | 0.2400 |
| 7 | 2 | 2 | 4 | 1 | B | 0.500 | -0.500 | 0.2500 |
| 10 | 1 | 0 | 1 | 1 | A | 1.000 | 0.000 | 0.0000 |

(Censoring at $t=8,9$ reduces risk sets but contributes no row since $d_j=0$ there.)

$$
\sum (O_{Aj}-E_{Aj}) = -0.500+0.429-0.500+0.400-0.500+0 = -0.671, \qquad \sum V_j = 1.2349
$$

$$
Z = \frac{-0.671}{\sqrt{1.2349}} = \frac{-0.671}{1.1113} = -0.604, \qquad Z^2 = 0.365.
$$

With $\chi^2_1$ reference, $p\approx0.55$: no evidence of a survival difference between groups, unsurprising given the tiny sample.

## Exercises

### Exercise 1

For $T\sim Exponential(\lambda)$ (constant hazard $h(t)=\lambda$), derive the hazard, survival, and cumulative hazard functions, and use them to prove the memoryless property $P(T>s+t\mid T>s) = P(T>t)$ directly from the constant-hazard assumption (i.e., show that constant hazard *implies* memorylessness, not just that the exponential distribution happens to have both properties).

<details>
<summary>Solution</summary>

If $h(t)=\lambda$ (constant) for all $t\geq0$, then $H(t) = \int_0^t \lambda\,du = \lambda t$, and therefore $S(t) = \exp(-H(t)) = e^{-\lambda t}$ (recovering the exponential survival function; density $f(t)=h(t)S(t)=\lambda e^{-\lambda t}$).

To show memorylessness follows from constant hazard, use the conditional survival identity: for $s,t\geq0$,

$$
P(T>s+t\mid T>s) = \frac{P(T>s+t, T>s)}{P(T>s)} = \frac{P(T>s+t)}{P(T>s)} = \frac{S(s+t)}{S(s)},
$$

since $\{T>s+t\}\subseteq\{T>s\}$. Substituting the constant-hazard survival function,

$$
\frac{S(s+t)}{S(s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = S(t) = P(T>t).
$$

So $P(T>s+t\mid T>s) = P(T>t)$ for all $s,t\geq 0$: having survived to time $s$ gives no information about additional survival time beyond $t$, the memoryless property, and it follows purely from $h(t)$ being constant (the hazard, "instantaneous risk given survival so far", literally does not depend on how long one has already survived, which is the mechanistic reason memorylessness holds). Conversely, one can show any absolutely continuous memoryless distribution must have $S(s+t)=S(s)S(t)$, whose only continuous solution is $S(t)=e^{-\lambda t}$ for some $\lambda\geq0$, so constant hazard and memorylessness are in fact equivalent characterizations of the exponential family among continuous distributions.

</details>

### Exercise 2

Derive Greenwood's formula via the delta method, starting from $\log\hat S(t) = \sum_{j:t_{(j)}\leq t}\log(1-d_j/n_j)$ and treating $d_j\mid n_j$ as approximately $Binomial(n_j,\lambda_j)$ with the $d_j$'s (conditionally on their risk sets) approximately independent across $j$.

<details>
<summary>Solution</summary>

Write $\hat\lambda_j = d_j/n_j$ and $\log\hat S(t) = \sum_{j} \log(1-\hat\lambda_j)$, sum over $j$ with $t_{(j)}\leq t$. Treating $d_j \mid n_j \sim Binomial(n_j,\lambda_j)$ (approximately, conditional on the risk-set structure) and the terms across different $j$ as approximately independent (a standard martingale-based argument makes this rigorous, but the binomial-approximation route gives the same classical formula), we have

$$
Var(\hat\lambda_j) \approx \frac{\lambda_j(1-\lambda_j)}{n_j},
$$

the usual binomial-proportion variance. Apply the delta method to $g(\hat\lambda_j) = \log(1-\hat\lambda_j)$, with $g'(\lambda_j) = -1/(1-\lambda_j)$:

$$
Var\big(\log(1-\hat\lambda_j)\big) \approx \big[g'(\lambda_j)\big]^2 Var(\hat\lambda_j) = \frac{1}{(1-\lambda_j)^2}\cdot\frac{\lambda_j(1-\lambda_j)}{n_j} = \frac{\lambda_j}{n_j(1-\lambda_j)}.
$$

Substituting the plug-in estimate $\hat\lambda_j = d_j/n_j$ (so $1-\hat\lambda_j = (n_j-d_j)/n_j$):

$$
\widehat{Var}\big(\log(1-\hat\lambda_j)\big) = \frac{d_j/n_j}{n_j\cdot (n_j-d_j)/n_j} = \frac{d_j}{n_j(n_j-d_j)}.
$$

By the assumed independence across event times, variances of the log-terms add:

$$
\widehat{Var}\big(\log\hat S(t)\big) = \sum_{j:t_{(j)}\leq t} \frac{d_j}{n_j(n_j-d_j)}.
$$

Finally, apply the delta method once more to transform back from $\log S$ to $S$: with $S = \exp(\log S)$, $\frac{dS}{d(\log S)} = S$, so

$$
\widehat{Var}\big(\hat S(t)\big) \approx \hat S(t)^2 \cdot \widehat{Var}\big(\log\hat S(t)\big) = \hat S(t)^2 \sum_{j:t_{(j)}\leq t} \frac{d_j}{n_j(n_j-d_j)},
$$

which is exactly Greenwood's formula.

</details>

### Exercise 3

Show that the log-rank test statistic is the score test for $H_0:\beta=0$ in a Cox proportional hazards model with a single binary covariate $x\in\{0,1\}$ indicating group membership. That is, derive the score $U(\beta)$ and expected information $I(\beta)$ of the Cox partial likelihood at $\beta=0$, and show $U(0)^2/I(0)$ matches the log-rank $Z^2$ statistic.

<details>
<summary>Solution</summary>

The Cox partial log-likelihood is $\ell(\beta) = \sum_{j=1}^k \left[x_{(j)}\beta - \log\sum_{l\in R(t_{(j)})} e^{x_l\beta}\right]$ for a single covariate. The score is

$$
U(\beta) = \frac{d\ell}{d\beta} = \sum_{j=1}^k \left[x_{(j)} - \frac{\sum_{l\in R(t_{(j)})} x_l e^{x_l\beta}}{\sum_{l\in R(t_{(j)})}e^{x_l\beta}}\right].
$$

At $\beta=0$, $e^{x_l\beta}=1$ for all $l$, so the weighted average collapses to a simple average over the risk set:

$$
\frac{\sum_{l\in R(t_{(j)})}x_l}{|R(t_{(j)})|} = \frac{n_{1j}}{n_j},
$$

since $x_l\in\{0,1\}$ and $\sum_l x_l = n_{1j}$ (number of group-1 subjects at risk), $|R(t_{(j)})|=n_j$. Also $x_{(j)}=1$ if the subject who failed at $t_{(j)}$ is in group 1, else $0$; that is, $x_{(j)}$ is the observed count $O_{1j}\in\{0,1\}$ (since exactly one event per distinct time in the no-ties case) and $n_{1j}/n_j = E_{1j}/d_j$ with $d_j=1$. Hence

$$
U(0) = \sum_{j=1}^k \big(O_{1j}-E_{1j}\big),
$$

exactly the log-rank numerator.

The observed/expected information at $\beta=0$ is $-\ell''(\beta)$ evaluated at $\beta=0$; a standard calculation for the Cox partial likelihood gives, at each risk set, a "weighted variance" term

$$
-\frac{d^2\ell_j}{d\beta^2}\bigg|_{\beta=0} = \frac{\sum_l x_l^2}{n_j} - \left(\frac{\sum_l x_l}{n_j}\right)^2 = \frac{n_{1j}}{n_j}-\left(\frac{n_{1j}}{n_j}\right)^2 = \frac{n_{1j}}{n_j}\cdot\frac{n_{2j}}{n_j} = \frac{n_{1j}n_{2j}}{n_j^2}
$$

(using $x_l^2=x_l$ for binary $x_l$). Summing over event times gives $I(0) = \sum_j n_{1j}n_{2j}/n_j^2$. This differs from the log-rank $V_j = n_{1j}n_{2j}d_j(n_j-d_j)/[n_j^2(n_j-1)]$ only by the finite-population correction factor $d_j(n_j-d_j)/(n_j-1)$, which equals $1$ when $d_j=1$ (the no-ties case used throughout this lesson's examples), since then $d_j(n_j-d_j)/(n_j-1) = 1\cdot(n_j-1)/(n_j-1)=1$. So under no ties, $I(0) = \sum_j V_j$ exactly, and

$$
\frac{U(0)^2}{I(0)} = \frac{\left(\sum_j(O_{1j}-E_{1j})\right)^2}{\sum_j V_j} = Z^2_{\text{log-rank}},
$$

confirming that the log-rank test is precisely the Cox partial-likelihood score test for $\beta=0$ with a single binary covariate (with ties, the two differ by the finite-population correction, which is why software sometimes reports "Score test" and "Log-rank test" as extremely close but not bit-identical numbers when there are tied event times).

</details>

### Exercise 4

For a new sample of $n=6$ subjects with times/status (1=event, 0=censored): $2(1), 3(1), 3(0), 5(1), 7(0), 8(1)$, compute the Kaplan-Meier estimate $\hat S(5)$, its Greenwood variance, and an approximate 95% confidence interval using the normal approximation.

<details>
<summary>Solution</summary>

Order the data: $t=2$ (event), $t=3$ (1 event + 1 censored, tie), $t=5$ (event), $t=7$ (censored), $t=8$ (event).

**Risk sets and KM factors** (convention: at a tied time, events are counted against the risk set before removing same-time censored observations):

- $t=2$: $n_1=6$, $d_1=1$ (event) $\Rightarrow$ factor $= 5/6$.
- $t=3$: $n_2=5$ (one subject removed after $t=2$), $d_2=1$ (one event; the co-occurring censored observation does not reduce the risk set for computing this factor, but both the event and the censoring reduce the risk set for later times) $\Rightarrow$ factor $=4/5$; risk set drops by $d_2 + (\text{censored at }t=3) = 1+1=2$, so $n_3 = 5-2=3$.
- $t=5$: $n_3=3$, $d_3=1$ $\Rightarrow$ factor $=2/3$.

$$
\hat S(2) = 5/6 = 0.8333, \qquad \hat S(3) = 0.8333\times4/5 = 0.6667, \qquad \hat S(5) = 0.6667\times2/3 = 0.4444.
$$

**Greenwood variance at $t=5$:**

$$
\sum_{j:t_{(j)}\leq5} \frac{d_j}{n_j(n_j-d_j)} = \frac{1}{6\times5}+\frac{1}{5\times4}+\frac{1}{3\times2} = 0.03333+0.05+0.16667 = 0.25000.
$$

$$
\widehat{Var}(\hat S(5)) = 0.4444^2 \times 0.25 = 0.19753\times0.25 = 0.049383, \qquad SE = \sqrt{0.049383} = 0.2222.
$$

**95% CI (normal approximation):**

$$
\hat S(5) \pm 1.96\times SE = 0.4444 \pm 1.96(0.2222) = 0.4444 \pm 0.4355 = (0.0089,\ 0.8799).
$$

This interval is very wide, reflecting the small risk sets late in follow-up (only 3 subjects at risk by $t=5$), and it comes close to the boundary at 0, a sign that with even slightly different data the naive normal-approximation interval could exceed $[0,1]$. In practice one would instead compute the interval on the $\log(-\log \hat S)$ scale and back-transform to guarantee the endpoints stay within $[0,1]$; the simple normal-approximation interval computed here is adequate pedagogically but would not be the production-quality choice for such a small risk set.

</details>
