```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Random-Feature & Deep Classifiers

## Motivation

Around 2020 a method appeared that trains in seconds, has no learned convolutions at all, and beat
almost everything on the UCR archive. ROCKET is worth understanding not only because you should be
running it as a baseline, but because *why* it works tells you what time-series classification
actually requires — and how little of it is deep learning.

## ROCKET: random convolutions + a linear model

Generate $k\approx10{,}000$ random 1D convolution kernels. Each kernel has a random length
(7, 9, 11), random weights $\sim\mathcal N(0,1)$ (mean-centred), a random bias, a random dilation
$d\sim 2^{\,\mathcal U(0,\log_2\frac{n-1}{l-1})}$ and random padding. Convolve each with the series
and extract **two** features per kernel:

$$
\max_i (x * \omega)_i \qquad\text{and}\qquad \mathrm{PPV}=\frac{1}{m}\sum_i \mathbb 1\big[(x*\omega)_i>0\big].
$$

Feed the $2k$ features to a ridge classifier (or logistic regression for large $N$). That is the
entire method: **nothing is learned except the linear layer**.

Why it works:

* Random dilations sample many *time scales* at once — the single most important design element,
  and the reason a fixed-size kernel bank covers both fast wiggles and slow trends.
* **PPV** — the proportion of positive values — is the crucial feature: it measures *how much* of
  the series matches a pattern, not just whether the best match was strong. Max-pooling alone
  performs markedly worse.
* Ten thousand random features span a rich function space; a regularised linear model on top is
  convex, has one hyperparameter, and cannot overfit the way a deep net can on 200 training series.

**MiniROCKET** makes it almost deterministic: a fixed set of two-valued kernels, only bias and
dilation varied, PPV only — roughly 30x faster with equal accuracy. It should be your default.
**MultiROCKET** adds three more pooling statistics and first-difference channels for a further
accuracy bump at a moderate cost. All handle multivariate input by convolving over a random subset
of channels.

Practical caveats: features are not interpretable individually; scale features before ridge; the
representation is huge ($2k$ columns), so watch memory on long multivariate collections; and the
`num_kernels` vs. accuracy curve saturates — 10k is convention, not magic.

## Deep classifiers

**InceptionTime** — an ensemble of 5 Inception-style 1D CNNs, each stacking modules with parallel
convolutions of several kernel sizes (10, 20, 40) plus a bottleneck and residual connections. The
multi-scale idea is the same one dilation gives ROCKET. It is the strongest deep classifier in the
bake-offs, and the ensemble is not optional: single models have high variance.

**HIVE-COTE 2.0** — a heterogeneous ensemble (shapelets, dictionary, interval and convolution
transforms) that tops accuracy tables and costs orders of magnitude more compute. Cite it as the
accuracy ceiling; deploy it rarely.

**Shapelets** — a *concept* worth keeping: a shapelet is a short subsequence whose presence
anywhere in the series discriminates classes, and shapelet methods search for the most informative
ones. Modern implementations are fast, but the reason to know shapelets is the mental model:
"local pattern, position-invariant" is a genuinely different inductive bias from both global
distance (DTW) and global statistics (features), and it is often the right description of a
maneuver signature.

## The bake-off literature, and how to read it

The UCR/UEA archive plus the bake-off papers (Bagnall et al. 2017; Ruiz et al. 2021; Middlehurst
et al. 2023) evaluate dozens of algorithms over >100 datasets with the same protocol, reporting
critical-difference diagrams from Friedman + post-hoc tests. The stable findings:

* 1NN-DTW is a genuinely strong baseline that a large fraction of published methods do not beat.
* ROCKET/MiniROCKET sit statistically level with far more complex methods at a tiny fraction of the
  cost.
* Deep models win when data is plentiful, labels are clean, and invariances are large; they lose on
  the small-$N$ datasets that dominate the archive (many have <100 training series).
* Differences between the top handful of methods are usually not statistically significant, which
  is exactly what a critical-difference diagram is designed to show you.

Read the diagram, not the table: a method one rank higher with overlapping cliques is not better.
And note the archive's own bias — mostly univariate, equal-length, pre-segmented, clean series.
Your trajectory data is none of those, so treat the ranking as a prior, not as a result.

## Suggested order of attack for a new problem

1. Kinematic features + random forest (Ch.0 L02) — hours, and it may end the project.
2. MiniROCKET + ridge — minutes, usually the top-2 answer.
3. 1NN-DTW with a tuned band — the honest distance-based reference.
4. InceptionTime — only if 1-3 leave a gap you can attribute to something specific, and you have
   the data to support it.

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| Equal-length, aligned series | trajectories of varying duration | padding artifacts dominate the random features | resample to fixed length or use feature/DTW families |
| Random features cover the signal | very long series, very rare local patterns | ROCKET underperforms a shapelet or interval method | longer kernels/dilations, MultiROCKET, segment first |
| Deep model needs no baseline | small $N$ | beaten by ridge on random features | always run steps 1-2 first |
| Benchmark ranking transfers | your data is messy/imbalanced/multivariate | ranking flips completely | re-run the comparison on *your* protocol |
| Accuracy is the metric | class imbalance, unequal error costs | useless model with high accuracy | per-class F1, cost-weighted metrics (Ch.5) |

**Lens check:** lens 1 (random convolutions are a *representation* — that is the whole trick) and
lens 2 (bake-off methodology, critical-difference diagrams, statistical vs. practical significance).

## Available Challenges

[Challenge 01 - Classify Trajectory Types](../challenges/challenge1_classify_trajectories.ipynb)
