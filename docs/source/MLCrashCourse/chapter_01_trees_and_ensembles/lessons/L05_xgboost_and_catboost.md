```
Author: Cfir Hadar

Tags: Done
```
# Lesson 05 - XGBoost and CatBoost

## Motivation

Gradient boosting as described in the previous lesson is a *framework*. The libraries that dominate tabular ML — XGBoost, LightGBM, CatBoost — are engineering + a few genuine algorithmic ideas on top of it. This lesson covers the two ideas you should actually understand: XGBoost's second-order objective, and CatBoost's solution to the categorical-feature problem.

## XGBoost: Second-Order Boosting with Explicit Regularization

XGBoost (Chen & Guestrin, 2016) replaces "fit a tree to the negative gradient" with a **second-order Taylor expansion** of the loss around the current model. With $g_i=\partial_{F}L(y_i,F(x_i))$ and $h_i=\partial^2_{F}L(y_i,F(x_i))$, the stage-$m$ objective for a candidate tree $f$ is

$$
\mathcal{L}^{(m)}\approx\sum_{i=1}^{n}\left[g_i f(x_i)+\frac{1}{2}h_i f^2(x_i)\right]+\Omega(f),
\qquad
\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2,
$$

where the tree has $T$ leaves with values $w_1,\dots,w_T$. Because a tree is piecewise-constant, the sum splits by leaf. With $G_j=\sum_{i\in leaf_j}g_i$ and $H_j=\sum_{i\in leaf_j}h_i$, minimizing over $w_j$ is a one-dimensional quadratic:

$$
w_j^*=-\frac{G_j}{H_j+\lambda},
\qquad
\mathcal{L}^*=-\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}+\gamma T.
$$

This closed form is the whole trick — it gives a **split gain formula**:

$$
\text{Gain}=\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma,
$$

used exactly where CART uses impurity decrease. Notice the structure: Newton's method instead of plain gradient descent (curvature $h_i$ weights the residuals), $\ell_2$ shrinkage $\lambda$ on leaf values, and a per-leaf cost $\gamma$ — the same $\ell_0$ complexity penalty we met in CART's cost-complexity pruning, but applied *during* growth (a split is only kept if its gain exceeds $\gamma$).

The rest of XGBoost's advantage is systems work worth knowing by name: histogram-based approximate split finding (bucket features into ~256 quantile bins → gradient statistics per bin), sparsity-aware "default directions" for missing values, column subsampling, parallelized split search, cache-aware layout. LightGBM pushed the histogram idea further (leaf-wise growth, gradient-based one-side sampling); conceptually it is the same model.

## The Categorical-Feature Problem

Trees split on thresholds, so categorical features must be encoded. One-hot works for low cardinality but explodes for high-cardinality features (user IDs, cities). The tempting alternative is **target encoding**: replace category $c$ with the mean target of that category,

$$
\text{enc}(c)=\frac{\sum_{i}\mathbb{1}\{x_i=c\}\,y_i+a\,p}{\sum_i \mathbb{1}\{x_i=c\}+a}
$$

(smoothed toward the prior $p$ with strength $a$, so rare categories aren't taken at face value). The problem: sample $i$'s own label $y_i$ participates in its own feature value. That is **target leakage** — the feature "knows" the answer, training metrics look great, and the model collapses at inference. Leakage is worst exactly for rare categories, where one label dominates the average.

## CatBoost: Ordered Target Statistics (briefly, but precisely)

CatBoost (Prokhorenkova et al., 2018) fixes the leakage with a beautifully simple device: **impose a random permutation $\sigma$ of the training samples and encode each sample using only the samples that precede it**:

$$
\text{enc}(x_i=c)=\frac{\sum_{j:\,\sigma(j)<\sigma(i)}\mathbb{1}\{x_j=c\}\,y_j+a\,p}{\sum_{j:\,\sigma(j)<\sigma(i)}\mathbb{1}\{x_j=c\}+a}.
$$

Think of it as processing the data as a *stream in random order*: each sample's encoding is an honest out-of-sample statistic, as if that sample had just arrived. No $y_i$ ever leaks into $x_i$. Several permutations are used across boosting iterations to reduce the variance this introduces. CatBoost also builds *combinations* of categorical features greedily (splitting on "city × device" interactions).

The same "use only the past" principle is applied to the boosting itself (**ordered boosting**): the residual of sample $i$ is computed by a model trained without sample $i$'s influence, removing a subtle bias in standard gradient boosting where residuals are computed on already-seen data. CatBoost additionally uses *oblivious trees* (the same split condition on every node of a level), which makes trees very fast to evaluate and acts as extra regularization.

## Practical guidance

* Default choice for tabular problems: **gradient boosting** (any of the three), with early stopping on a validation set. Random forest as the no-tuning baseline.
* Many/strong categorical features → **CatBoost** first.
* Big data, need speed → **LightGBM**.
* Tune, in order of impact: number of trees (via early stopping), learning rate, depth / number of leaves, subsampling ratios, $\lambda$/$\gamma$-style regularizers.

## Walkthrough

[Walkthrough - Trees & Ensembles in Practice](../walkthroughs/lesson_trees_ensembles.ipynb)

## Available Challenges

[Challenge 01 - Tabular Ensembles](../challenges/challenge1_tabular_ensembles.ipynb)
