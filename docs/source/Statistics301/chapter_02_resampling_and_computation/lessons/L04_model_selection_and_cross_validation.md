# Model Selection & Cross-Validation

## Motivation

Statistics 1/2 compared nested models using $F$-tests and $R^2$/adjusted $R^2$ -- tools that
answer a narrow question within one model class, not how to choose among many candidates, tune
hyperparameters, or estimate performance on new data. This lesson develops cross-validation as
an estimator of out-of-sample error, its bias-variance behavior, information criteria (AIC,
BIC, Mallows' $C_p$), and the inferential pitfalls of using the same data to select and evaluate
a model.

## The Prediction-Error Target and Bias-Variance of CV

For a new $(X,Y)$ independent of training data $\mathcal D=\{(X_i,Y_i)\}_{i=1}^n$ and fit
$\hat f_{\mathcal D}$, the target is $\text{Err}=E_{(X,Y),\mathcal D}[L(Y,\hat f_{\mathcal
D}(X))]$. $K$-fold CV partitions $\mathcal D$ into folds $\mathcal D_1,\dots,\mathcal D_K$, fits
$\hat f_{\mathcal D_{-k}}$ on all but fold $k$, and averages held-out loss:

$$
\widehat{\text{Err}}_{CV} = \frac1n\sum_{k=1}^K\sum_{i\in\mathcal D_k} L\big(Y_i,\hat f_{\mathcal D_{-k}}(X_i)\big).
$$

This targets $E_{\mathcal D}[\text{Err}(\mathcal D)]$ for training size $n(K-1)/K$, not the
error of the model trained on all $n$ points.

**Bias**: training on $n(K-1)/K<n$ observations generally biases $\widehat{\text{Err}}_{CV}$
upward relative to the full-data model (learning curves typically decrease in training size);
bias shrinks as $K\to n$.

**Variance**: the $K$ training sets overlap (for $K>2$, any two share $n(K-2)/K$ points),
inducing positive correlation among fold errors. The naive variance estimator
$\frac1n\sum_i(\ell_i-\bar\ell)^2$ (treating per-observation losses as independent) understates
the true variance of $\widehat{\text{Err}}_{CV}$; no generally unbiased variance estimator
exists for $K$-fold CV (Bengio & Grandvalet, 2004).

Small $K$ gives higher bias but often lower variance (less-correlated, more different folds);
large $K$ gives lower bias but folds are nearly identical to the full fit (highly correlated),
which can inflate variance for unstable learners.

## K-Fold vs. Leave-One-Out

**LOO** is $K=n$: $\widehat{\text{Err}}_{LOO}=\frac1n\sum_i L(Y_i,\hat f_{\mathcal D_{-i}}(X_i))$.
LOO has minimal bias (folds train on $n-1$ points) but the $n$ fitted models are extremely
similar, so the $n$ terms are highly correlated -- averaging many correlated quantities reduces
variance less effectively than averaging fewer, more independent ones. This can make LOO a
higher-variance estimator of $\text{Err}$ than 5-/10-fold CV for unstable learners, though the
effect is data- and model-dependent.

For linear smoothers ($\hat Y=HY$), LOO has a closed-form shortcut avoiding $n$ refits:

$$
\widehat{\text{Err}}_{LOO} = \frac1n\sum_i \left(\frac{Y_i-\hat Y_i}{1-H_{ii}}\right)^2,
$$

with $H_{ii}$ the leverage and $\hat Y_i$ from the full-data fit -- the PRESS statistic (summed),
making LOO free for linear models though costly ($n$ refits) for general nonlinear ones. $K=5$
or $10$ is the standard default elsewhere, balancing bias/variance at $K$ (not $n$) refits.

## Nested CV for Hyperparameter Selection

Using one round of $K$-fold CV to both select a hyperparameter (e.g. ridge $\lambda$) and report
that model's error is double-dipping: the reported error is a minimum over a search evaluated on
the same data, biasing it downward (optimistic) -- analogous to reporting the smallest of
several p-values as a single prespecified test.

**Nested CV** separates the roles: an **outer loop** ($K_{outer}$ folds) held out purely for
final performance estimation; within each outer-training fold, an **inner loop**
($K_{inner}$-fold CV) selects $\hat\lambda_k$ without touching the outer test fold; refit on the
full outer-training data with $\hat\lambda_k$ and evaluate once on the outer test fold; average
across $k$. This estimates the error of the *entire selection-plus-fitting pipeline*, not of any
single fixed $\lambda$ -- more expensive ($K_{outer}\times K_{inner}$ fits per candidate) but
the standard remedy when both a trustworthy estimate and a hyperparameter choice are needed.

## AIC and BIC

**AIC.** For MLE $\hat\theta$ with $p$ parameters, a Taylor expansion of expected KL discrepancy
using asymptotic normality of the MLE gives

$$
E_{new}[-2\log f(\text{new}\mid\hat\theta)] \approx -2\ell(\hat\theta) + 2p,
$$

i.e. in-sample deviance underestimates expected out-of-sample deviance by $\approx2p$ ("optimism"),
motivating $\text{AIC}=-2\ell(\hat\theta)+2p$ (smaller preferred). AIC estimates expected
predictive deviance and, under general conditions, is asymptotically equivalent to LOO-CV for
the same model class -- a fast closed-form approximation avoiding refits.

**BIC.** A Laplace approximation to the marginal likelihood $P(\text{data}\mid M_k)=\int
f(\text{data}\mid\theta)\pi(\theta)\,d\theta$ gives, to leading order,
$\log P(\text{data}\mid M_k)\approx\ell(\hat\theta)-\frac p2\log n$, so

$$
\text{BIC} = -2\ell(\hat\theta) + p\log n.
$$

BIC's penalty grows with $n$ (approximating the log Bayes factor), unlike AIC's fixed $2p$.

**When each applies.** AIC is not model-selection-consistent -- it asymptotically overfits with
positive probability, favoring larger models even as $n\to\infty$. BIC is consistent when the
true model has fixed finite dimension: $P(\text{BIC selects truth})\to1$. Use AIC for predictive
goals with no assumed true finite model; use BIC when a sparse true model is a reasonable
working assumption. Since $p\log n>2p$ whenever $n>e^2\approx7.39$, BIC favors smaller models
than AIC at essentially every practical sample size.

## Mallows' $C_p$

For linear regression, squared error, Gaussian errors with known/estimated $\sigma^2$:

$$
C_p = \frac{SSE_p}{\hat\sigma^2_{full}} - n + 2p,
$$

using $\hat\sigma^2_{full}$ from the largest model. $C_p\approx p$ indicates negligible bias;
$C_p\gg p$ indicates underfitting. Under Gaussian errors with known $\sigma^2$, $C_p$ is
algebraically equivalent to AIC up to a constant/rescaling -- $C_p$ is the classical
linear-regression-specific version, AIC the general likelihood-based one.

## Selection Inference Caveat: Post-Selection Inference

When a model is *selected* using the data (stepwise selection, best-subset by AIC/BIC, CV-tuned
$\lambda$), standard errors and p-values computed as if that model were fixed in advance are
invalid -- the **post-selection inference** problem. Selecting predictors by in-sample
significance and reporting naive p-values for survivors ignores the extra variability from
selection: only predictors that happened to look significant in this sample were retained,
analogous to reporting the minimum of several p-values uncorrected.

Valid remedies: (a) **sample splitting** -- select on one part, infer on an independent held-out
part; (b) specialized **post-selection inference** (e.g. the polyhedral/selective-inference
framework for LASSO, conditioning on the selection event); (c) treat the entire
selection-plus-fitting pipeline as the object of study (as nested CV does for prediction error).
This applies regardless of whether selection uses a p-value threshold, AIC/BIC, or CV.

```python
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
def nested_cv_score(estimator, param_grid, X, y, k_outer=5, k_inner=5):
    outer = KFold(n_splits=k_outer, shuffle=True, random_state=0)
    inner = KFold(n_splits=k_inner, shuffle=True, random_state=1)
    search = GridSearchCV(estimator, param_grid, cv=inner)
    return cross_val_score(search, X, y, cv=outer)
```

## Worked Example

$n=100$, four nested models $p=2,3,4,5$, $SSE_p=220,195,190,188$, $\hat\sigma^2_{full}=1.90$.

$$
C_p(2)=115.79-100+4=19.79,\quad C_p(3)=8.63,\quad C_p(4)=8.00,\quad C_p(5)=8.95.
$$

$p=4$ has the lowest $C_p$ ($\approx p$, negligible bias); $p=2$'s $C_p\gg p$ signals
underfitting. With $\text{AIC}(p)\approx n\log(SSE_p/n)+2p$, $\text{BIC}(p)\approx
n\log(SSE_p/n)+p\log n$, and $\log100=4.605$:

| $p$ | $n\log(SSE_p/n)$ | AIC | BIC |
|---|---|---|---|
| 2 | 78.85 | 82.85 | 88.06 |
| 3 | 66.78 | 72.78 | 80.59 |
| 4 | 64.19 | 72.19 | 82.60 |
| 5 | 63.13 | 73.13 | 86.14 |

AIC is minimized at $p=4$ (72.19, barely below $p=3$), agreeing with $C_p$. BIC is minimized at
$p=3$ (80.59) -- a smaller model, exactly as expected since BIC's penalty ($4.605p$) is much
steeper than AIC's ($2p$) at $n=100$.

## Exercises

### Exercise 1

For OLS with hat matrix $H=X(X^TX)^{-1}X^T$, prove
$Y_i-\hat Y_i^{(-i)} = (Y_i-\hat Y_i)/(1-H_{ii})$, where $\hat Y_i^{(-i)}$ is the leave-$i$-out
prediction. (Sherman-Morrison may be used without re-derivation.)

<details>
<summary>Solution</summary>

Let $x_i^T$ be row $i$ of $X$, $A=(X^TX)^{-1}$. Since $X_{(-i)}^TX_{(-i)}=X^TX-x_ix_i^T$,
Sherman-Morrison gives

$$
(X_{(-i)}^TX_{(-i)})^{-1} = A + \frac{Ax_ix_i^TA}{1-H_{ii}}, \qquad H_{ii}=x_i^TAx_i.
$$

With $X_{(-i)}^TY_{(-i)}=X^TY-x_iY_i$,

$$
\hat\beta^{(-i)} = \left[A+\frac{Ax_ix_i^TA}{1-H_{ii}}\right](X^TY-x_iY_i).
$$

Using $AX^TY=\hat\beta$, $x_i^TAX^TY=\hat Y_i$, $x_i^TAx_i=H_{ii}$, expand and collect:

$$
\hat\beta^{(-i)} = \hat\beta - Ax_i\left[Y_i - \frac{\hat Y_i}{1-H_{ii}} + \frac{H_{ii}Y_i}{1-H_{ii}}\right] = \hat\beta - \frac{Ax_i(Y_i-\hat Y_i)}{1-H_{ii}},
$$

since $-Y_i-H_{ii}Y_i/(1-H_{ii}) = -Y_i/(1-H_{ii})$ combines with $+\hat Y_i/(1-H_{ii})$ to give
$(\hat Y_i-Y_i)/(1-H_{ii})$. Then $\hat Y_i^{(-i)}=x_i^T\hat\beta^{(-i)} = \hat Y_i -
\frac{H_{ii}(Y_i-\hat Y_i)}{1-H_{ii}}$, so

$$
Y_i-\hat Y_i^{(-i)} = (Y_i-\hat Y_i)\left[1+\frac{H_{ii}}{1-H_{ii}}\right] = \frac{Y_i-\hat Y_i}{1-H_{ii}}. \qquad \blacksquare
$$

</details>

### Exercise 2

For nested models $M_0$ ($p_0$ params) and $M_1$ ($p_1=p_0+1$), show AIC prefers $M_1$ iff the
likelihood-ratio statistic exceeds 2, and relate this to AIC's implied significance level versus
the conventional $\chi^2_{1,0.95}\approx3.84$.

<details>
<summary>Solution</summary>

$\text{AIC}(M_1)<\text{AIC}(M_0)$ iff $-2\ell(\hat\theta_1)+2p_1<-2\ell(\hat\theta_0)+2p_0$, i.e.
(using $p_1-p_0=1$) $2[\ell(\hat\theta_1)-\ell(\hat\theta_0)]>2$. The left side is exactly the
LR statistic $\Lambda$ for $H_0:\beta_{extra}=0$, so AIC prefers $M_1$ iff $\Lambda>2$.

Under $H_0$, $\Lambda\sim\chi^2_1$ asymptotically. A conventional $\alpha=0.05$ LR test rejects
at $\Lambda>3.84$; AIC's threshold of 2 is much lower, so AIC will select $M_1$ in cases where a
5% test would retain $M_0$. AIC's implied significance level is $P(\chi^2_1>2)\approx0.157$ --
about 15.7%, far more liberal than 5%, and this level never shrinks with $n$ (fixed penalty),
explaining why AIC tends to overfit asymptotically. By contrast BIC's implied level
$P(\chi^2_1>\log n)\to0$ as $n\to\infty$, giving its consistency.

</details>

### Exercise 3

With 20 noise predictors, $n=50$, and best-subset selection over all $2^{20}$ models by CV
error, explain via a thought experiment why reporting standard OLS p-values for the selected
model is invalid, and describe the sample-splitting fix and its cost.

<details>
<summary>Solution</summary>

If all 20 predictors are pure noise (true coefficients all 0), comparing $\approx10^6$ candidate
subsets means some subset will, by pure sampling variability in this one $n=50$ dataset, achieve
spuriously low CV error and get selected; OLS on that selected subset will typically show
"significant" p-values despite every true coefficient being zero. This is the multiple-comparisons
problem: the reported p-values assume a fixed, non-random design, but the design was chosen
using $Y$, so the true sampling distribution of the selected coefficients (accounting for the
search) has much fatter tails than the naive reference distribution, making the true Type I
error rate far exceed 5%.

**Fix**: randomly split the 50 observations into a selection set and an inference set. Select
the subset using only the selection set; fit OLS for that now-fixed subset using only the
inference set. Since inference-set data played no role in choosing predictors, standard OLS
inference is exactly valid conditional on the (independently chosen) model.

**Cost**: data inefficiency -- each half does only one job, so both selection (noisier with
fewer points) and inference (wider CIs, lower power) suffer relative to using all 50 for each
task. With small original $n$ and many candidates, this loss can be severe, motivating
selective-inference alternatives that avoid discarding data at the cost of model-specific
technical machinery.

</details>

### Exercise 4

If the $K$ fold-level errors $\bar e_1,\dots,\bar e_K$ were iid with variance $\tau^2$, show the
naive estimator $\frac1{K(K-1)}\sum_k(\bar e_k-\widehat{\text{Err}}_{CV})^2$ is unbiased for
$\text{Var}(\widehat{\text{Err}}_{CV})$; then explain why independence fails in practice and the
direction of the resulting bias.

<details>
<summary>Solution</summary>

If iid, $\text{Var}(\widehat{\text{Err}}_{CV})=\text{Var}(\frac1K\sum_k\bar e_k)=\tau^2/K$. The
sample variance $S^2=\frac1{K-1}\sum_k(\bar e_k-\bar e)^2$ is the standard unbiased estimator of
$\tau^2$, so $S^2/K = \frac1{K(K-1)}\sum_k(\bar e_k-\widehat{\text{Err}}_{CV})^2$ is unbiased for
$\tau^2/K$ -- exactly the claim, by the ordinary unbiasedness of sample variance applied to the
$K$ fold averages treated as iid.

**Why independence fails**: for $K>2$, training sets $\mathcal D_{-k}$ and $\mathcal D_{-k'}$
share $n(K-2)/K$ observations, so $\hat f_{\mathcal D_{-k}}$ and $\hat f_{\mathcal D_{-k'}}$ are
both close to the same full-data fit -- when that fit is unusually good or bad for this
particular realized dataset, all $K$ fold errors move together, inducing positive correlation.
With common pairwise correlation $\rho>0$,
$\text{Var}(\widehat{\text{Err}}_{CV}) = \frac{\tau^2}K[1+(K-1)\rho] > \tau^2/K$, but the naive
$S^2/K$ still targets roughly $\tau^2/K$ (it is not designed to detect correlation among the
very quantities it averages), so it **underestimates** the true variance whenever $\rho>0$ -- the
typical case due to fold overlap. This is the Bengio-Grandvalet result: no generally unbiased CV
variance estimator exists, and the common naive formula is anti-conservative in practice.

</details>
