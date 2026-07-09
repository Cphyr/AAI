```
Author: Cfir Hadar

Tags: Done
```
# Lesson 03 - Random Forests

## Motivation

Bagging's variance floor is $\rho\sigma^2$: bootstrap-trained trees are too similar to each other, because they mostly agree on which features are the strongest and put them at the root. If one feature dominates, *every* bagged tree splits on it first, and the trees are nearly clones. Random forests (Breiman, 2001) attack $\rho$ directly.

## The Algorithm

A random forest is bagging plus **feature subsampling at every split**:

1. For $b=1,\dots,B$: draw a bootstrap resample.
2. Grow a deep tree, but at *each node*, draw a random subset of $m$ out of $d$ features and pick the best split **only among those $m$**.
3. Average (regression) or vote (classification).

Defaults that work well: $m=\sqrt{d}$ for classification, $m=d/3$ for regression; trees grown deep with small leaves. Note that $m$ is the knob trading correlation against strength: smaller $m$ → less correlated trees (lower $\rho$) but individually weaker (higher $\sigma^2$, higher bias). Recall

$$
\text{Var}(\hat f_{RF})=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2,
$$

so the game is minimizing $\rho\sigma^2$; the optimum is usually at surprisingly small $m$. With $m=d$ a random forest degenerates to plain bagging.

Everything from the bagging lesson carries over: OOB error is free, and increasing $B$ never overfits — pick $B$ by "when the OOB curve flattens" (hundreds, typically).

## Why Randomization of Splits Works

The randomization means that even a dominant feature only appears at the root of $\sim m/d$ of the trees; the other trees are *forced* to explore alternative split structures. Individually they are worse; collectively they cover different aspects of the signal and their errors decorrelate. This is the same principle as dropout in deep networks, and more generally: **an ensemble gains exactly as much as its members disagree while still being right on average** (the "ambiguity decomposition": ensemble error = average member error − average member disagreement).

## Feature Importance

Two standard measures, both worth knowing along with their failure modes:

* **Mean Decrease in Impurity (MDI)**: total impurity decrease $\Delta$ attributed to each feature, averaged over trees. Computed for free during training. *Biased toward high-cardinality/continuous features* (they simply offer more candidate thresholds).
* **Permutation importance**: shuffle one feature's column in the OOB/validation data and measure the drop in score. Model-agnostic and less biased, but *splits importance between correlated features* (shuffling one is compensated by its correlated twin, so both look unimportant).

Use permutation importance on OOB data as the default; treat MDI as a quick diagnostic.

## Extremely Randomized Trees (ExtraTrees)

One step further: don't bootstrap, and instead of searching for the best threshold, draw a *random* threshold per candidate feature and pick the best among those random cuts. Even lower correlation, even higher bias, much faster training. Sometimes wins, always worth a try since it's one line of code (`ExtraTreesClassifier`).

## Where Random Forests Sit

* **Strengths**: near-zero tuning (defaults are good), robust to noise and outliers, no feature scaling, parallel training, built-in uncertainty via the spread of tree predictions, OOB validation for free.
* **Weaknesses**: bias of the *individual* deep tree remains — forests can't extrapolate beyond the range of training targets (predictions are averages of training-set values in leaves), and they lose to boosting on most well-specified tabular tasks because averaging can't reduce bias.

That last weakness — averaging attacks only variance — motivates the complementary strategy: build trees *sequentially*, each one correcting the bias left by its predecessors. That is boosting.

## Walkthrough

[Walkthrough - Trees & Ensembles in Practice](../walkthroughs/lesson_trees_ensembles.ipynb)
