```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Transformer Forecasters & the DLinear Lens

## Motivation

Between 2019 and 2022 a series of transformer forecasters claimed large gains on long-horizon
benchmarks. In 2022 a paper titled *"Are Transformers Effective for Time Series Forecasting?"*
showed that a **single linear layer** matched or beat them on the same benchmarks. The resulting
argument is the most instructive case study in the field about evaluation protocols, and it is the
real subject of this lesson.

## The architectures

**Informer (2021)** — attacks the $O(L^2)$ cost with ProbSparse attention (attend only from queries
with high attention entropy) plus distilling layers, and predicts the whole horizon in one forward
pass (a "generative" decoder). The lasting contribution is direct multi-step decoding, not the
sparse attention.

**Autoformer / FEDformer** — build seasonal-trend decomposition into the architecture, and replace
dot-product attention with auto-correlation or frequency-domain mixing. The decomposition, again,
turns out to carry much of the benefit.

**PatchTST (2023)** — two ideas, both simple:

1. **Patching**: split the series into (possibly overlapping) subsequences of length $P$ and embed
   each as one token. A 512-step input becomes 32 tokens instead of 512 — attention cost drops by
   $\sim P^2$, each token carries local semantics instead of a single scalar, and the effective
   context lengthens.
2. **Channel independence**: run the same univariate model over each channel, sharing weights,
   rather than mixing channels at the input. Fewer parameters, far less overfitting, and it often
   *improves* accuracy — a strong hint about how little cross-channel signal these benchmarks have.

![Point-wise tokens vs. patches](../../../_static/ts/patching.png)

**TFT (Temporal Fusion Transformer)** — the practitioner's model: explicitly separates static
covariates (platform type), known-future covariates (schedule, day of week) and observed past
covariates; variable-selection networks weight inputs per instance; gated residual connections;
quantile outputs (Ch.6 L01) rather than point predictions; interpretable attention over time. When
you have real covariate structure and need multi-horizon quantiles, this is often the right choice
regardless of the benchmark debate.

## DLinear, and the controversy

**DLinear** is: decompose $x$ into trend (moving average) and remainder, apply one linear layer
$\hat y = W x$ to each, sum. That is it — no nonlinearity, no attention. NLinear is even simpler:
subtract the last value, apply a linear layer, add it back. On the standard long-horizon benchmarks
(ETT, Electricity, Traffic, Weather) these matched or beat every transformer available at the time.

Why? Several reasons, and it is worth separating them:

* Those benchmarks are dominated by **trend and strong fixed seasonality** — structure a linear map
  over a long lookback captures completely.
* Attention is **permutation-equivariant**; positional encodings recover order imperfectly, which is
  a poor bias for series whose order is everything.
* Many reported gains came from **normalisation and protocol details** (instance normalisation,
  lookback length, whether the last value is subtracted) rather than architecture. When the linear
  baseline is given the same lookback and normalisation, the gap largely closes.
* Datasets are small relative to transformer capacity, and the standard train/valid/test split is a
  single chronological cut — one draw, no fold-level uncertainty (Ch.1 L03).

The follow-up literature (PatchTST, TimesNet, and later re-evaluations) shows transformers *can*
win once patching and proper normalisation are in place, on datasets with enough data and richer
structure. The honest summary is not "transformers are useless" but: **on these benchmarks, the
architecture explained less of the variance than the protocol did.**

## Strong simple baselines to keep in the running

* **N-BEATS** — deep stacks of fully-connected blocks with backcast/forecast residuals and basis
  expansion; interpretable (trend/seasonality) or generic. Still very competitive.
* **N-HiTS** — N-BEATS plus multi-rate sampling and hierarchical interpolation: much cheaper for
  long horizons, and typically better.
* **Linear/DLinear/NLinear** — must be in every comparison table you produce. They cost seconds.

## How to read (and run) a forecasting comparison

1. Is the **lookback length** tuned per method, or fixed to the winner's preference? Linear models
   improve monotonically with longer lookbacks; several transformers do not.
2. Is **normalisation** identical across methods? Instance normalisation alone can move the ranking.
3. Is the split a single chronological cut, or rolling origin with fold-level results?
4. Are results averaged over **seeds**? Deep models on small datasets have seed variance comparable
   to the reported gains.
5. Is there a **statistical test** on the difference (Ch.1 L03)?
6. Are the **naive baselines** (seasonal naive, last value, CV extrapolation) in the table at all?
7. Does the metric match the decision, and is it computed on normalised or original scale?

If the answer to three of these is unfavourable, the ranking in the paper tells you about the
protocol, not about the models. The Chapter 7 walkthrough makes you do this yourself: you will flip
the ranking of the same models by changing the protocol alone.

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| Benchmark rank transfers to your data | different structure, sampling, horizon | reproduced model loses to a linear baseline | re-benchmark on your protocol |
| Architecture explains the gain | normalisation/lookback confound | ablation moves accuracy more than the model does | forced ablation (walkthrough) |
| Long lookback always helps | drift, regime changes | longer context hurts | tune lookback per method |
| Channel mixing helps | weak cross-channel signal | overfitting, worse than channel-independent | test channel independence |
| One test split is enough | short, nonstationary series | ranking flips with the cutoff | rolling origin + fold distribution |

**Lens check:** lens 2, thoroughly — this lesson is a case study in evaluation-protocol
sensitivity — plus lens 1 (patching is a representation choice).

## Walkthrough

[DLinear vs. Patch Models vs. a Classical Baseline](../walkthroughs/lesson_dlinear_vs_patch.ipynb)

## Next

[Lesson 03 - Trajectory Prediction Proper](L03_trajectory_prediction.md)
