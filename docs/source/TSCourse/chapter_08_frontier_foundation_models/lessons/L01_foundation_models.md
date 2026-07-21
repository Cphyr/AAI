```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - Foundation Models for Time Series

## Motivation

Language and vision got foundation models; time series is trying. The attempt is interesting
precisely because time series lacks what made those work: there is no shared vocabulary across
domains, no semantic unit like a word, no agreed unit or scale, and no reason a heart-rate trace and
an exchange rate should share structure. This lesson is a map of the current attempts and of the
methodological trouble they are in.

## The vocabulary problem

A token in text means the same thing in every document. A value of "37.2" means body temperature,
degrees of bank, or a stock price depending entirely on context — and the sampling rate, units and
scale differ per dataset. Every TS foundation model is essentially an answer to: *what is the token,
and how do we make series comparable?*

## Three answers

**Chronos (Amazon, 2024).** Scale each series (mean absolute scaling), then **quantise** values into
a fixed vocabulary of bins and train an off-the-shelf language-model architecture (T5) with
cross-entropy on the next token. Forecasts are produced by autoregressive sampling, giving a
predictive *distribution* for free. Trained on a large public corpus plus synthetic data (Gaussian
processes, composed patterns). Trade-offs: quantisation caps resolution and cannot represent values
outside the trained range; autoregressive decoding is slow at long horizons; but zero-shot accuracy
is genuinely competitive, and the simplicity of the recipe is the point.

**TimesFM (Google, 2024).** Decoder-only transformer over **patched** inputs (Ch.7 L02), trained on
~100B time points from real and synthetic sources, with an output patch longer than the input patch
so long horizons need few decoding steps. Point forecasts by default; quantile heads available.

**Moirai (Salesforce, 2024).** Encoder-only, masked-prediction training, explicitly **any-variate**:
flattens multivariate series into one sequence with variate identifiers, and uses multiple patch
sizes to handle different frequencies. Outputs a mixture distribution, so uncertainty is native.

Others in the same wave: Lag-Llama (lag features + LLaMA architecture), TimeGPT (commercial),
Moment and TimesNet (representation-oriented). The design axes that actually distinguish them:
*tokenisation* (quantised values / patches / lags), *architecture* (encoder / decoder / enc-dec),
*probabilistic output* (sampling / quantiles / mixture), and *training corpus* (which is where most
of the difference lives).

## Self-supervised objectives

* **Masked modeling** — mask spans and reconstruct (BERT-style). Strong for representation learning;
  note that masking *spans* is essential, since masking single points is trivially solved by
  interpolation.
* **Contrastive** (TS2Vec, TF-C) — pull augmented views of the same window together, push others
  apart. Everything depends on the augmentations, and the standard ones (jitter, scaling, cropping,
  permutation) are not all label-preserving for trajectories: permuting segments destroys exactly
  the maneuver ordering you care about.
* **Forecasting as pretext** — predict the next window; the most natural fit, and what most
  foundation models actually use.

## Zero-shot: what to believe

Reported zero-shot results are often close to, sometimes better than, task-specific models on
standard benchmarks. Before extrapolating that to your problem:

* **Benchmark leakage is a live, unresolved issue.** These corpora are assembled from public
  archives, and the evaluation sets come from the same archives. "Zero-shot" on a dataset that may
  sit in the pretraining corpus is not zero-shot. Some papers document decontamination; many do
  not. This is the field's current methodological weak point, and you should ask about it first.
* **Baselines.** Check the naive baselines are present (Ch.0 L02). Several zero-shot claims shrink
  substantially against seasonal naive and a tuned linear model.
* **Domain fit.** These models are trained mostly on energy, retail, traffic and web data. A
  maneuvering-target trajectory has nothing in common with those, and the useful prior may be
  close to zero.
* **Cost.** A 200M-parameter model per forecast, versus a Kalman filter. For streaming track data
  the compute argument alone often ends the discussion.

Where they genuinely help: cold start (no history for a new series), many heterogeneous series with
no budget to model each, quick strong baselines during exploration, and as feature extractors for
downstream tasks.

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| Pretraining corpus resembles your data | specialised domains (tracks, sensors) | zero-shot loses to CV extrapolation | fine-tune, or do not use it |
| Zero-shot means unseen | benchmark contamination | implausibly good published numbers | evaluate on *your* private data |
| Scaling/quantisation is lossless | wide dynamic range, sharp transients | resolution loss, clipped extremes | check the reconstruction of the tokenisation itself |
| One model for all frequencies | mixed sampling rates | poor handling of your rate | check the patch/frequency handling; resample |
| Uncertainty is calibrated | sampled forecasts | over-confident intervals | PIT/coverage checks (Ch.6 L03) |

**Lens check:** lens 1 (tokenisation is representation) and lens 2 (leakage and baselines in
benchmark claims).

## Next

[Lesson 02 - State-Space Models as Deep Architectures](L02_deep_ssms.md)
