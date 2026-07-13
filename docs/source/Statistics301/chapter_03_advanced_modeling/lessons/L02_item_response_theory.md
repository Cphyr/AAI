# Item Response Theory (IRT)

## Motivation

Classical test theory summarizes a test-taker's performance by a raw or percent-correct score, treating all items as interchangeable and ignoring that items differ in difficulty and discriminating power. Item response theory instead posits a **latent trait** (ability) underlying observed item responses and models each item's response probability as a function of that trait. This yields item-invariant person measurement (in principle, ability estimates do not depend on which particular items were administered), person-invariant item calibration, and a rigorous machinery for adaptive testing, differential item functioning detection, and test information. IRT models are also, as we will see, close relatives of the logistic mixed models from Lesson 1, which gives us an estimation toolkit we already partially understand.

## Latent Trait Models: Rasch, 2PL, and 3PL

### The latent trait framework

Let $\theta_j$ denote the latent ability of person $j$, and let $X_{ij} \in \{0,1\}$ be the response of person $j$ to item $i$ (correct/incorrect). The central modeling assumption is **local independence**: conditional on $\theta_j$, the responses to different items are independent,

$$
P(X_{1j}=x_1,\dots,X_{Kj}=x_K \mid \theta_j) = \prod_{i=1}^K P(X_{ij}=x_i\mid\theta_j).
$$

The function $P_i(\theta) = P(X_i=1\mid\theta)$ is the **item characteristic curve** (ICC): a monotonically increasing, S-shaped function of ability.

### Rasch (1PL) model

$$
P_i(\theta) = P(X_{ij}=1\mid\theta_j,b_i) = \frac{\exp(\theta_j-b_i)}{1+\exp(\theta_j-b_i)},
$$

where $b_i$ is the item's **difficulty** (the ability level at which $P_i=0.5$). The Rasch model has a distinguishing mathematical property: the total raw score $\sum_i X_{ij}$ is a **sufficient statistic** for $\theta_j$ (proved in Exercise 2), and person and item parameters are "separable" in the sense that conditional maximum likelihood can estimate item parameters without knowing any $\theta_j$.

### 2PL model

$$
P_i(\theta) = \frac{\exp\big(a_i(\theta-b_i)\big)}{1+\exp\big(a_i(\theta-b_i)\big)},
$$

adding a **discrimination** parameter $a_i>0$ controlling the steepness of the ICC at $\theta=b_i$. Larger $a_i$ means the item more sharply distinguishes examinees just below vs. just above $b_i$; raw score is no longer sufficient for $\theta$ once items differ in $a_i$.

### 3PL model

$$
P_i(\theta) = c_i + (1-c_i)\,\frac{\exp\big(a_i(\theta-b_i)\big)}{1+\exp\big(a_i(\theta-b_i)\big)},
$$

adding a **guessing** (pseudo-guessing) parameter $c_i\in[0,1)$, a lower asymptote representing the chance of a correct response from an examinee with vanishingly low ability (relevant for multiple-choice items). The ICC now ranges from $c_i$ to $1$ rather than from $0$ to $1$.

## Item and Test Information

Fisher information quantifies how precisely an item (or test) can pin down $\theta$. For the 2PL model,

$$
I_i(\theta) = a_i^2\, P_i(\theta)\big(1-P_i(\theta)\big),
$$

maximized at $\theta = b_i$ where $P_i=0.5$, and increasing in $a_i^2$: highly discriminating items concentrated near a test-taker's ability level are maximally informative. For the 3PL model, the lower asymptote attenuates information:

$$
I_i(\theta) = a_i^2\,\frac{\big(P_i(\theta)-c_i\big)^2}{(1-c_i)^2}\cdot\frac{1-P_i(\theta)}{P_i(\theta)}.
$$

Because local independence makes the log-likelihood additive across items, Fisher information is additive too:

$$
I(\theta) = \sum_{i=1}^K I_i(\theta), \qquad SE(\hat\theta) \approx \frac{1}{\sqrt{I(\theta)}}.
$$

This is the basis of computerized adaptive testing: at each step, administer the item that maximizes information at the current ability estimate.

## Estimation: Marginal ML and Ability Estimation

### Marginal maximum likelihood (integrating out ability)

When calibrating item parameters $\xi = \{a_i,b_i,c_i\}$ from a sample of $N$ examinees, treating each $\theta_j$ as a fixed unknown ("joint MLE") produces inconsistent item-parameter estimates as $N\to\infty$ with a fixed test length (the incidental-parameters problem: the number of $\theta_j$'s grows with $N$). The standard remedy is **marginal maximum likelihood** (MMLE): treat $\theta_j$ as random with a population distribution $g(\theta)$ (typically $N(0,1)$) and integrate it out,

$$
L(\xi) = \prod_{j=1}^N \int \prod_{i=1}^K P_i(\theta)^{x_{ij}}\big(1-P_i(\theta)\big)^{1-x_{ij}}\, g(\theta)\,d\theta.
$$

This marginal likelihood is maximized via the EM algorithm (Bock and Aitkin, 1981): the E-step approximates the integral by Gauss-Hermite quadrature and computes, for each quadrature node, the posterior-weighted expected number of examinees and correct responses; the M-step then updates $\{a_i,b_i,c_i\}$ item-by-item via weighted logistic regression on the pseudo-data from the E-step. This mirrors exactly the structure of estimating a generalized linear mixed model by integrating out a random intercept, which is the conceptual link developed below.

### Ability estimation: MLE and EAP

Once item parameters are calibrated (treated as known), a person's ability is estimated from their response pattern.

**MLE**: maximize $\ell(\theta) = \sum_i \big[x_i\log P_i(\theta) + (1-x_i)\log(1-P_i(\theta))\big]$ via Newton-Raphson,

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\sum_i (x_i - P_i(\theta^{(t)}))}{\sum_i P_i(\theta^{(t)})(1-P_i(\theta^{(t)}))} \quad \text{(Rasch case, } a_i=1\text{)}.
$$

MLE is undefined (diverges to $\pm\infty$) for all-correct or all-incorrect response patterns, since the likelihood is then monotonic in $\theta$ with no interior maximum.

**EAP** (expected a posteriori, Bayesian) avoids this by using a prior $g(\theta)$:

$$
\hat\theta_{EAP} = E[\theta\mid x] = \frac{\int \theta\, L(x\mid\theta)\, g(\theta)\, d\theta}{\int L(x\mid\theta)\, g(\theta)\, d\theta},
$$

computed by quadrature. Like the BLUP of Lesson 1, EAP shrinks extreme response patterns toward the prior mean and is always finite, at the cost of some bias toward the prior for examinees with genuinely extreme ability.

## DIF and the Link to Logistic Mixed Models

### Differential item functioning

An item shows **DIF** if examinees from different groups (e.g., defined by language or gender) with the *same underlying ability* have different probabilities of a correct response. **Uniform DIF** is a constant shift in difficulty $b_i$ across groups (ICCs are shifted but non-crossing); **non-uniform DIF** involves a group-by-ability interaction (e.g., differing $a_i$), so the ICCs cross. Detection methods include:

- **Mantel-Haenszel**: stratify examinees by observed total score (a proxy for $\theta$), form a 2x2 table of group x correct/incorrect within each stratum, and test whether the common odds ratio across strata equals 1.
- **IRT likelihood-ratio test**: fit the item's parameters constrained equal across groups vs. freely estimated per group, and compare by LRT.
- **Logistic regression**: $\text{logit}(P) = \beta_0 + \beta_1\theta + \beta_2 G + \beta_3(\theta\times G)$, where $G$ is group membership; $\beta_2\ne0$ signals uniform DIF, $\beta_3\ne0$ signals non-uniform DIF.

### Link to logistic mixed models

Rewrite the Rasch model with person ability treated as random rather than fixed:

$$
\text{logit}\, P(X_{ij}=1) = \theta_j - b_i, \qquad \theta_j \sim N(0,\sigma^2),
$$

with items as fixed effects (a factor with $K-1$ contrasts) and persons as a random intercept. This is *exactly* a logistic mixed model (a generalized linear mixed model with binomial response and random intercept per person), fittable with standard mixed-model software using a person random effect and an item fixed effect, e.g. `glmer(correct ~ item + (1|person), family=binomial)`. Marginal ML for the Rasch model, integrating over the person random effect, coincides with GLMM estimation via Gauss-Hermite quadrature. The 2PL model, by contrast, has $a_i\theta_j$, a *product* of an item parameter and the random effect, which behaves like a person-specific "random slope on a latent factor" and is not a standard GLMM; it requires either specialized IRT software or a nonlinear mixed-model formulation. This connection is why many applied papers describe the Rasch model as "a logistic mixed model with items as fixed effects and persons as random effects."

## Worked Example

Three items with Rasch difficulties $b = (-1, 0, 1)$; examinee with $\theta = 0.5$.

$$
P_1 = \frac{e^{0.5-(-1)}}{1+e^{1.5}} = \frac{e^{1.5}}{1+e^{1.5}} = \frac{4.4817}{5.4817} = 0.8176
$$

$$
P_2 = \frac{e^{0.5}}{1+e^{0.5}} = \frac{1.6487}{2.6487} = 0.6225, \qquad P_3 = \frac{e^{-0.5}}{1+e^{-0.5}} = \frac{0.6065}{1.6065} = 0.3775
$$

Item informations ($I_i = P_i(1-P_i)$ since $a_i=1$): $I_1 = 0.8176\times0.1824 = 0.1491$, $I_2 = 0.6225\times0.3775 = 0.2350$, $I_3 = 0.3775\times0.6225=0.2350$. Test information $I(0.5) = 0.6191$, so $SE(\hat\theta) = 1/\sqrt{0.6191} = 1.2708$.

Suppose the examinee's observed pattern is $x=(1,1,0)$ (correct on items 1-2, incorrect on 3). One Newton-Raphson step from $\theta^{(0)}=0.5$:

$$
\text{score} = (1-0.8176)+(1-0.6225)+(0-0.3775) = 0.1824+0.3775-0.3775 = 0.1824
$$

$$
\text{information} = I_1+I_2+I_3 = 0.6191
$$

$$
\theta^{(1)} = 0.5 + \frac{0.1824}{0.6191} = 0.5+0.2947 = 0.7947.
$$

This is identical in form to a Fisher-scoring step for logistic regression MLE, since the Rasch score/information equations are exactly the logistic-regression score/information equations with the $b_i$ acting as known offsets.

## Exercises

### Exercise 1

Derive the score function and Fisher information for MLE ability estimation in the Rasch model with $K$ items of known difficulties $b_1,\dots,b_K$, and show the resulting Newton-Raphson update is algebraically identical to a Fisher-scoring step for a logistic regression of $x_i$ on a constant, with $-b_i$ entered as a known offset.

<details>
<summary>Solution</summary>

The log-likelihood for a single examinee's pattern $x=(x_1,\dots,x_K)$ given ability $\theta$ is

$$
\ell(\theta) = \sum_{i=1}^K \Big[x_i\log P_i(\theta) + (1-x_i)\log(1-P_i(\theta))\Big], \qquad P_i(\theta) = \frac{e^{\theta-b_i}}{1+e^{\theta-b_i}}.
$$

Since $\frac{d}{d\theta}P_i(\theta) = P_i(\theta)(1-P_i(\theta))$ (standard logistic derivative), the score is

$$
U(\theta) = \frac{d\ell}{d\theta} = \sum_i \left[\frac{x_i}{P_i}-\frac{1-x_i}{1-P_i}\right]P_i(1-P_i) = \sum_i\big(x_i - P_i(\theta)\big).
$$

The observed information is $-\frac{d^2\ell}{d\theta^2} = \sum_i P_i(\theta)(1-P_i(\theta))$, which does not depend on $x_i$, so observed and expected (Fisher) information coincide: $I(\theta) = \sum_i P_i(1-P_i)$.

Newton-Raphson/Fisher scoring: $\theta^{(t+1)} = \theta^{(t)} + U(\theta^{(t)})/I(\theta^{(t)})$.

Now consider ordinary logistic regression of binary outcomes $x_i$ on an intercept-only model with a known offset $-b_i$: $\text{logit}(P_i) = \theta - b_i$ (i.e., $\theta$ is the only free parameter, $-b_i$ is a fixed offset per observation). The GLM score and Fisher information for the single parameter $\theta$ are, by the standard logistic regression formulas $U(\theta)=\sum_i (x_i-P_i)\cdot \partial \eta_i/\partial\theta$ and $I(\theta) = \sum_i P_i(1-P_i)(\partial\eta_i/\partial\theta)^2$ with $\partial\eta_i/\partial\theta=1$, exactly

$$
U(\theta) = \sum_i(x_i-P_i), \qquad I(\theta)=\sum_i P_i(1-P_i),
$$

identical to the Rasch ability score and information. Hence estimating a single examinee's Rasch ability by Newton-Raphson is literally fitting an intercept-only logistic regression with known item-difficulty offsets, and the update formulas coincide term for term.

</details>

### Exercise 2

Prove that in the Rasch model, the raw score $S = \sum_{i=1}^K X_i$ is a sufficient statistic for $\theta$ (fix item difficulties $b_1,\dots,b_K$ as known).

<details>
<summary>Solution</summary>

The joint probability of a response pattern $x=(x_1,\dots,x_K)$ under local independence is

$$
P(X=x\mid\theta) = \prod_{i=1}^K \left(\frac{e^{\theta-b_i}}{1+e^{\theta-b_i}}\right)^{x_i}\left(\frac{1}{1+e^{\theta-b_i}}\right)^{1-x_i} = \frac{\prod_i e^{x_i(\theta-b_i)}}{\prod_i(1+e^{\theta-b_i})}.
$$

Expand the numerator:

$$
\prod_i e^{x_i(\theta-b_i)} = \exp\left(\theta\sum_i x_i - \sum_i x_i b_i\right) = e^{\theta S}\, e^{-\sum_i x_ib_i}.
$$

So

$$
P(X=x\mid\theta) = \underbrace{e^{\theta S}}_{\text{depends on }x\text{ only through }S}\cdot \underbrace{e^{-\sum_i x_ib_i}}_{h(x),\ \text{free of }\theta}\cdot\underbrace{\Big[\prod_i(1+e^{\theta-b_i})\Big]^{-1}}_{c(\theta),\ \text{free of }x}.
$$

This is exactly the exponential-family factorization $P(X=x\mid\theta) = g(S,\theta)\,h(x)$ with $g(S,\theta) = e^{\theta S}c(\theta)$. By the Fisher-Neyman factorization theorem, $S=\sum_i X_i$ is a sufficient statistic for $\theta$. (Note this relies critically on the Rasch model's additive $\theta - b_i$ form with a common coefficient of $1$ on $\theta$ for every item; in the 2PL model the coefficient is $a_i$, so the exponent becomes $\theta\sum_i a_ix_i$, meaning the *weighted* score $\sum_i a_i x_i$, not the raw score, would be sufficient, and since the $a_i$ are generally unknown/estimated this sufficiency is not usable in practice, which is why raw-score sufficiency is considered a special, prized property of the Rasch model.)

</details>

### Exercise 3

Sketch the derivation of the EM algorithm's E-step for marginal maximum likelihood item calibration: derive the posterior density of $\theta_j$ given examinee $j$'s response pattern and current parameter estimates $\xi^{(t)}$, and explain how Gauss-Hermite quadrature turns this into a computationally tractable weighted-likelihood update.

<details>
<summary>Solution</summary>

By Bayes' theorem, the posterior density of $\theta_j$ given the response pattern $x_j$ and current item-parameter estimates $\xi^{(t)}$ is

$$
h(\theta \mid x_j, \xi^{(t)}) = \frac{\prod_{i=1}^K P_i(\theta;\xi_i^{(t)})^{x_{ij}}\big(1-P_i(\theta;\xi_i^{(t)})\big)^{1-x_{ij}}\; g(\theta)}{\int \prod_{i=1}^K P_i(\theta';\xi_i^{(t)})^{x_{ij}}\big(1-P_i(\theta';\xi_i^{(t)})\big)^{1-x_{ij}}\; g(\theta')\, d\theta'},
$$

i.e. the likelihood of the observed pattern at ability $\theta$, times the population prior $g(\theta)$, normalized. This follows directly from Bayes' rule applied to the joint density of $(\theta_j, x_j)$ implied by the MMLE model (person ability random with density $g$, response pattern conditionally generated by the IRT model given $\theta_j$).

The complete-data log-likelihood (if $\theta_j$ were observed) would be $\sum_j \sum_i \big[x_{ij}\log P_i(\theta_j)+(1-x_{ij})\log(1-P_i(\theta_j))\big]$, linear in indicator-type sufficient statistics for each item. The E-step replaces this by its expectation under $h(\theta\mid x_j,\xi^{(t)})$ for each examinee, i.e., needs $E_{h}[\log P_i(\theta)]$-type terms, which requires evaluating $\int \log P_i(\theta) \, h(\theta\mid x_j,\xi^{(t)})\, d\theta$. Since this integral has no closed form for logistic ICCs, **Gauss-Hermite quadrature** approximates it (and the normalizing integral) by a finite weighted sum over fixed nodes $\theta_1^*,\dots,\theta_Q^*$ with quadrature weights $w_1,\dots,w_Q$ chosen to integrate polynomials (times a Gaussian kernel) exactly:

$$
\int f(\theta)g(\theta)\,d\theta \approx \sum_{q=1}^Q w_q\, f(\theta_q^*).
$$

Concretely, for each examinee $j$ and each node $q$, compute the likelihood $L(x_j\mid\theta_q^*,\xi^{(t)})$, form the discretized posterior weight $\pi_{jq} \propto w_q L(x_j\mid\theta_q^*,\xi^{(t)})\,g(\theta_q^*)$ (normalized over $q$), and accumulate, for each item $i$, the "expected number of examinees at node $q$" and "expected number correct at node $q$" by summing $\pi_{jq}$ and $\pi_{jq}x_{ij}$ over $j$. The M-step then treats these accumulated pseudo-counts at each of the $Q$ ability nodes as binomial data and updates each item's parameters $\xi_i$ by (weighted) logistic regression of expected-correct-count on the quadrature-node abilities. Iterating E and M steps converges to the MMLE item parameter estimates, exactly analogous to the EM algorithm used for GLMMs with a random intercept integrated out via quadrature.

</details>

### Exercise 4

An item is examined for uniform DIF between a Reference group and a Focal group using the Mantel-Haenszel procedure with two score strata. The 2x2 counts (correct/incorrect) are:

Stratum 1 (n=55): Reference correct = 20, Reference incorrect = 10; Focal correct = 10, Focal incorrect = 15.
Stratum 2 (n=58): Reference correct = 25, Reference incorrect = 5; Focal correct = 20, Focal incorrect = 8.

Compute the Mantel-Haenszel common odds ratio

$$
OR_{MH} = \frac{\sum_k n_{R1k}n_{F0k}/n_{Tk}}{\sum_k n_{F1k}n_{R0k}/n_{Tk}}
$$

and interpret the result for DIF.

<details>
<summary>Solution</summary>

Label within stratum $k$: $n_{R1k}$ = reference correct, $n_{R0k}$ = reference incorrect, $n_{F1k}$ = focal correct, $n_{F0k}$ = focal incorrect, $n_{Tk}$ = stratum total.

**Stratum 1** ($n_{T1}=30+25=55$): $n_{R1,1}=20$, $n_{F0,1}=15$, $n_{F1,1}=10$, $n_{R0,1}=10$.
- Numerator term: $20\times15/55 = 300/55 = 5.4545$
- Denominator term: $10\times10/55 = 100/55 = 1.8182$

**Stratum 2** ($n_{T2}=30+28=58$): $n_{R1,2}=25$, $n_{F0,2}=8$, $n_{F1,2}=20$, $n_{R0,2}=5$.
- Numerator term: $25\times8/58 = 200/58 = 3.4483$
- Denominator term: $20\times5/58 = 100/58 = 1.7241$

Summing:

$$
\sum_k n_{R1k}n_{F0k}/n_{Tk} = 5.4545+3.4483 = 8.9028
$$

$$
\sum_k n_{F1k}n_{R0k}/n_{Tk} = 1.8182+1.7241 = 3.5423
$$

$$
OR_{MH} = \frac{8.9028}{3.5423} = 2.513.
$$

Interpretation: after stratifying on ability (approximated by score group), reference-group examinees have about $2.5\times$ the odds of answering the item correctly compared to focal-group examinees at the same ability level. Since $OR_{MH}$ is far from 1 (a common rule of thumb, the ETS classification, flags $|\Delta MH| $ corresponding to $OR_{MH}$ outside roughly $(0.53, 1.9)$ or wider bounds as at least moderate DIF), this item shows evidence of uniform DIF favoring the reference group and would be a candidate for review or removal from the operational item bank.

</details>
