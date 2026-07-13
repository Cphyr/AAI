# Multiple Testing & FDR

## Motivation

Statistics 2 taught you to control the Type I error of a *single* test. Applied work rarely
involves a single test: A/B testing platforms evaluate dozens of metrics per experiment, genomics
studies test thousands of genes. If you run $m$ independent tests at $\alpha=0.05$ under the
global null, $P(\text{at least one false rejection}) = 1-(1-\alpha)^m$ exceeds 50% by around
$m=14$. This lesson develops the two dominant error-control frameworks -- family-wise error rate
(FWER) and false discovery rate (FDR) -- and the classical procedures (Bonferroni, Holm,
Benjamini-Hochberg) used to control them, closing with A/B testing and genomics applications.

## FWER versus FDR

Let $m$ nulls be tested, $m_0\leq m$ true, $V$ = false rejections, $R$ = total rejections.

**FWER** $=P(V\geq1)$: probability of *any* false rejection. Controlling FWER at $\alpha$ is a
strong guarantee -- with probability $1-\alpha$, zero false discoveries across the whole family.
Appropriate when even one false positive is costly.

**FDR** $=E[V/\max(R,1)]$: expected *proportion* of rejections that are false. Controlling FDR at
$q$ allows some false discoveries as long as they are a bounded fraction of all discoveries --
appropriate for exploratory, high-volume settings where a modest contamination rate buys far
greater power. When all nulls are true, $V=R$ whenever $R>0$, so FDR reduces exactly to FWER
(Exercise 2); when many nulls are false, FDR can be far smaller, which is why FDR-controlling
procedures reject far more hypotheses at the same nominal level.

## FWER Procedures: Bonferroni and Holm

**Bonferroni.** Reject $H_i$ if $p_i\leq\alpha/m$. By the union bound,
$\mathrm{FWER}=P(\bigcup_{i\in\mathcal I_0}\{p_i\leq\alpha/m\}) \leq \sum_{i\in\mathcal I_0}
\alpha/m \leq m_0\alpha/m \leq \alpha$, valid under *any* dependence.

**Holm (step-down).** Order $p_{(1)}\leq\dots\leq p_{(m)}$. Find the smallest $k$ with
$p_{(k)}>\alpha/(m-k+1)$; reject $H_{(1)},\dots,H_{(k-1)}$. Since thresholds $\alpha/(m-k+1)\geq
\alpha/m$ for every $k$, Holm rejects at least as much as Bonferroni (Exercise 1) while still
controlling FWER at $\alpha$ under arbitrary dependence: the calibrated threshold sequence lets
the same union-bound argument apply to the true nulls alone, regardless of where false nulls'
p-values fall.

## Benjamini-Hochberg and FDR Control

**BH at level $q$.** Order $p_{(1)}\leq\dots\leq p_{(m)}$; find the largest $k$ with

$$
p_{(k)} \leq \frac{k}{m}q,
$$

reject $H_{(1)},\dots,H_{(k)}$.

**Proof intuition under independence.** For a true null $H_i$, conditional on the data-dependent
threshold $\frac{R}{m}q$ (with $R$ the realized rejection count), independence and uniformity of
true-null p-values give $E\left[\mathbb 1[p_i\leq\frac{R}{m}q]/\max(R,1)\right] \leq q/m$ for each
true null. Summing over the $m_0$ true nulls,

$$
\mathrm{FDR} = \sum_{i\in\mathcal I_0} E\left[\frac{\mathbb 1[i\text{ rejected}]}{\max(R,1)}\right] \leq m_0\cdot\frac{q}{m} \leq q.
$$

Each true null contributes at most $q/m$; summing over at most $m$ of them caps FDR at $q$. Since
the threshold $\frac{k}{m}q$ grows linearly with rank (vs. fixed $\alpha/m$ for Bonferroni or
slowly-relaxing Holm thresholds), BH rejects substantially more when many nulls are truly false.

## Q-values and Dependence Caveats

A **q-value** is the minimum FDR level at which a hypothesis would be rejected by BH: $\hat
q_{(i)} = \min_{j\geq i}\frac{m}{j}p_{(j)}$ (running minimum for monotonicity), the FDR analogue of
a p-value.

**Dependence caveats.** BH's proof assumed independence; it also holds under *positive regression
dependence*. Under arbitrary (including negative) dependence it can fail, and the
**Benjamini-Yekutieli (BY)** correction uses

$$
p_{(k)} \leq \frac{k}{m\,c(m)}q, \qquad c(m)=\sum_{j=1}^m\frac1j \approx \ln m + 0.577,
$$

controlling FDR under any dependence at the cost of far fewer rejections (Exercise 4). BH is used
by default; BY is reserved for suspected adversarial/negative dependence.

## Applications: A/B Testing and Genomics

**A/B testing.** Platforms report p-values for many metrics per experiment simultaneously
(conversion, revenue, retention, segment cuts). Treating each in isolation at $\alpha=0.05$
inflates the platform-wide false-positive rate; standard fixes are BH across reported metrics
within an experiment, or Holm when the decision is binary "ship or don't" and one false positive
could trigger a costly launch. Sequential peeking before a fixed sample size is a related but
distinct problem (testing the same hypothesis repeatedly over time), requiring separate
corrections (alpha-spending), not covered by BH/Holm directly.

**Genomics.** A study testing $m=20{,}000$ genes at $\alpha=0.05$ uncorrected would expect
$\sim1{,}000$ false positives under the global null. Bonferroni/Holm require $p<2.5\times10^{-6}$
per gene, often leaving too few discoveries for an exploratory screen. BH at $q=0.05$ is the
field-standard compromise: among "significant" genes, on average no more than 5% are expected
false, while retaining useful power. Q-values are typically reported alongside p-values.

## Worked Example

Five p-values $p=(0.001,0.008,0.015,0.04,0.20)$, $m=5$.

**Bonferroni** ($\alpha/5=0.01$): reject $0.001,0.008$ only ($0.015>0.01$).

**Holm**: thresholds $\alpha/(m-i+1)$ for $i=1..5$: $0.01,0.0125,0.0167,0.025,0.05$. Compare
sequentially: $0.001\leq0.01$ reject; $0.008\leq0.0125$ reject; $0.015\leq0.0167$ reject;
$0.04>0.025$ stop. Rejects 3 -- one more than Bonferroni.

**BH** ($q=0.05$): thresholds $\frac{k}{5}(0.05)$: $0.01,0.02,0.03,0.04,0.05$. Check: all of
$0.001,0.008,0.015,0.04$ are $\leq$ their threshold ($0.04\leq0.04$ exactly); $0.20>0.05$ fails.
Largest valid $k=4$, so BH rejects 4 -- illustrating Bonferroni $\subseteq$ Holm $\subseteq$ BH.

```python
import numpy as np
p = np.array([0.001, 0.008, 0.015, 0.04, 0.20])
m, alpha = len(p), 0.05

bonf_reject = (p <= alpha / m).sum()

order = np.argsort(p)
holm_thresh = alpha / (m - np.arange(m))
holm_reject = 0
for i in range(m):
    if p[order][i] <= holm_thresh[i]:
        holm_reject += 1
    else:
        break

bh_thresh = (np.arange(1, m + 1) / m) * alpha
below = p[order] <= bh_thresh
bh_reject = (np.max(np.where(below)[0]) + 1) if below.any() else 0

print(bonf_reject, holm_reject, bh_reject)  # 2 3 4
```

## Exercises

### Exercise 1

Prove Holm rejects a superset of Bonferroni's rejections: if $p_i\leq\alpha/m$, Holm also rejects
$H_i$.

<details>
<summary>Solution</summary>

Let $p_i=p_{(r)}$ for rank $r$. Since $p_{(1)}\leq\dots\leq p_{(r)}\leq\alpha/m$, every $p_{(j)}$
for $j\leq r$ satisfies $p_{(j)}\leq\alpha/m$. Holm's threshold at position $j$ is
$\alpha/(m-j+1)\geq\alpha/m$ (since $m-j+1\leq m$). So for $j\leq r$, $p_{(j)}\leq\alpha/m\leq
\alpha/(m-j+1)$: no failure can occur at or before position $r$. Holm's first-failure index $k$
must therefore satisfy $k>r$, so Holm rejects at least $H_{(1)},\dots,H_{(r)}$, including
$H_{(r)}=H_i$. The worked example shows the containment can be strict (Holm rejected 3,
Bonferroni 2).

</details>

### Exercise 2

Prove that under the global null ($m_0=m$), FDR control at level $q$ is exactly equivalent to
FWER control at level $q$, for any procedure.

<details>
<summary>Solution</summary>

Under the global null every rejection is false, so $R=V$ deterministically. Then $R/\max(R,1) =
\mathbb 1[R\geq1]$ exactly: if $R=0$ the ratio is $0$; if $R\geq1$, $\max(R,1)=R$ so the ratio is
$1$. Thus

$$
\mathrm{FDR} = E\left[\frac{V}{\max(R,1)}\right] = E\left[\frac{R}{\max(R,1)}\right] = E[\mathbb 1[R\geq1]] = P(R\geq1) = P(V\geq1) = \mathrm{FWER}.
$$

FDR and FWER are literally the same quantity under the global null for any procedure, so
controlling one at $q$ controls the other at $q$. Practical differences (BH rejecting more than
Holm/Bonferroni) only emerge when $m_1=m-m_0>0$, since then $R>V$ and $V/R$ can be much smaller
than $\mathbb 1[R\geq1]$.

</details>

### Exercise 3

A study tests $m=10{,}000$ genes: $m_0=9{,}500$ true nulls (assume negligible false rejections
among them), $m_1=500$ true signals each detected with power $0.6$. A realization gives $V=15$
false rejections and the expected number of true rejections. Compute $R$ and the realized FDP
$V/R$, compare to target $q=0.05$, and comment on whether a single realization's FDP matching $q$
is meaningful.

<details>
<summary>Solution</summary>

True rejections: $0.6\times500=300$. So $R=300+15=315$, and $V/R = 15/315\approx0.0476$, close to
but not identical to $q=0.05$.

**Comment.** FDR control is a statement about $E[V/\max(R,1)]$ averaged over hypothetical repeated
realizations of the entire experiment, not a guarantee about the false discovery proportion in any
single realized dataset. In one realization, $V$ (and $V/R$) is random and can deviate from $q$
by chance, exactly analogous to how a 95% confidence interval does not guarantee 95% coverage in a
single realized interval -- it is a property of the procedure, not of one outcome. A q-value or
FDR target should always be understood as a long-run procedural guarantee.

</details>

### Exercise 4

Sketch why the Benjamini-Yekutieli correction factor $c(m)=\sum_{j=1}^m 1/j$ restores FDR control
under arbitrary dependence, and why BY is more conservative than BH.

<details>
<summary>Solution</summary>

BH's independence-based proof bounds each true null's contribution to $E[V/\max(R,1)]$ by exactly
$q/m$. Under arbitrary dependence this per-null bound can fail: correlated p-values break the
"each true null contributes independently" argument. A general Boole-type inequality for
arbitrarily dependent variables shows that, in the worst case, the bound that holds exactly under
independence picks up a multiplicative penalty $c(m)=\sum_{j=1}^m1/j$, giving
$\mathrm{FDR}\leq\frac{m_0}{m}c(m)q\leq c(m)q$ under the unmodified BH threshold. Replacing $q$
with $q/c(m)$ -- i.e. the BY threshold $p_{(k)}\leq\frac{k}{m\,c(m)}q$ -- brings this back down to
exactly $q$, valid under any dependence structure.

**Why more conservative.** $c(m)\approx\ln m + 0.577$ grows without bound (e.g. $c(10{,}000)
\approx9.79$), so the BY threshold is roughly ten-fold stricter than BH's for $m=10{,}000$, and
the gap widens with $m$. This is the price of making no assumption about dependence, versus BH's
un-penalized guarantee which relies on independence or positive regression dependence.

</details>
