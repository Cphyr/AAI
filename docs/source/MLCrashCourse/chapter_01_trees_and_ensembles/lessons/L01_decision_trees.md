```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - Decision Trees

## Motivation

Most real-world business data is tabular: rows of samples, columns of heterogeneous features (numeric, ordinal, categorical). On such data, tree-based models are still the strongest baseline — often beating deep networks — while being fast to train and easy to interpret. Everything in this chapter (bagging, random forests, boosting, XGBoost, CatBoost) is built on top of a single primitive: the decision tree. So let's understand it properly.

## The Model

A decision tree partitions the input space $\mathbb{R}^d$ into $M$ disjoint axis-aligned regions $R_1,\dots,R_M$ (the leaves) and predicts a constant in each region:

$$
f(x)=\sum_{m=1}^{M}c_m\cdot\mathbb{1}\{x\in R_m\}.
$$

Each internal node tests a single feature against a threshold ($x_j\le t$), so the regions are hyper-rectangles.

![https://upload.wikimedia.org/wikipedia/commons/e/eb/Decision_Tree.jpg](https://upload.wikimedia.org/wikipedia/commons/e/eb/Decision_Tree.jpg)

Finding the *optimal* partition is NP-complete, so all practical algorithms grow the tree **greedily**: at each node, pick the single best split, recurse on the two children, and never look back.

## Choosing a Split: Impurity

Let node $t$ hold a sample set with class proportions $p_1,\dots,p_K$. We measure how "mixed" the node is with an impurity function $H(t)$:

* **Gini index** (CART's default): $\quad H_{Gini}(t)=\sum_{k=1}^{K}p_k(1-p_k)=1-\sum_k p_k^2$
* **Entropy** (ID3/C4.5): $\quad H_{Ent}(t)=-\sum_{k=1}^{K}p_k\log p_k$

Both are concave, maximized at the uniform distribution and zero for a pure node. A candidate split $s$ sends $n_L$ samples to child $t_L$ and $n_R$ to $t_R$; its quality is the **impurity decrease**

$$
\Delta(s,t)=H(t)-\frac{n_L}{n}H(t_L)-\frac{n_R}{n}H(t_R),
$$

and we greedily choose $s^*=\arg\max_s \Delta(s,t)$. With entropy, $\Delta$ is exactly the **information gain** — the mutual information between the split indicator and the label. Concavity of $H$ guarantees $\Delta\ge 0$ (Jensen's inequality), i.e., splitting never *looks* harmful, which is exactly why trees overfit if you let them grow.

For **regression trees**, the impurity is the in-node variance, $H(t)=\frac{1}{n_t}\sum_{i\in t}(y_i-\bar{y}_t)^2$, and the leaf prediction is the mean $c_m=\bar{y}_{R_m}$ (the minimizer of squared loss). With absolute loss the optimal leaf value is the median — trees adapt to the loss through the leaf estimate and the impurity.

Complexity: for each node we sort each feature once and scan thresholds, giving $O(d\,n\log n)$ per level — cheap enough to build thousands of trees, which is where ensembles come in.

## Controlling Complexity

A fully grown tree has zero training error and huge variance: change a few samples and the greedy split near the root flips, changing the *entire* subtree below it. Trees are **low-bias, high-variance** estimators. Standard controls:

1. **Pre-pruning**: `max_depth`, `min_samples_leaf`, `min_impurity_decrease`.
2. **Post-pruning** (CART's cost-complexity pruning): grow fully, then minimize
   $$
   C_\alpha(T)=\sum_{m=1}^{|T|}n_m H(t_m)+\alpha|T|,
   $$
   where $|T|$ is the number of leaves. Sweeping $\alpha$ produces a nested sequence of subtrees; pick $\alpha$ by cross-validation. This is $\ell_0$ regularization on the number of leaves — remember this form, XGBoost's objective (Lesson 05) contains exactly such a $\gamma|T|$ term.

## The Tree Family (the gist)

You will mostly meet CART in practice (it's what sklearn, XGBoost and friends build on), but know the names:

| Algorithm | Splits | Impurity | Notes |
| --------- | ------ | -------- | ----- |
| ID3 | multi-way on categoricals | entropy | no numeric features, no pruning |
| C4.5 | multi-way | gain *ratio* | normalizes gain by split entropy — fixes ID3's bias toward high-cardinality features |
| CART | always binary | Gini / variance | handles regression, cost-complexity pruning |
| Oblique trees | linear combinations $w^Tx\le t$ | any | non-axis-aligned boundaries, rarely worth the extra variance |

Two properties worth remembering: trees are **invariant to monotone transformations** of features (only the order matters — no need to scale/normalize), and they handle mixed feature types and missing values gracefully (via surrogate splits or default directions).

## Why a Single Tree Is Not Enough

The greedy construction makes a single tree unstable: a small perturbation of the training set can produce a completely different tree. In bias-variance language, the estimator has low bias but very high variance. The next lessons are all one idea: **keep the low bias, kill the variance by averaging many trees** — the only question is how to make the trees different from each other (bagging, random forests) or how to make them correct each other (boosting).

## Walkthrough

[Walkthrough - Trees & Ensembles in Practice](../walkthroughs/lesson_trees_ensembles.ipynb)
