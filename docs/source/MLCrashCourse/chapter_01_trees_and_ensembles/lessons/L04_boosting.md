```
Author: Cfir Hadar

Tags: Done
```
# Lesson 04 - Boosting

## Motivation

Bagging and random forests train trees *in parallel* on the same task and average away variance; the bias stays. Boosting flips the picture: train trees *sequentially*, each one fitting what the current ensemble still gets wrong. The result is an additive model

$$
F_M(x)=\sum_{m=1}^{M}\eta\, f_m(x),
$$

built from **weak learners** — typically *shallow* trees (depth 3-6, sometimes stumps). Boosting is a bias-reduction technique; the weak learners individually barely beat chance, and the sequence turns them into a strong learner. This is not just a heuristic: the original question "can weak learnability be boosted to strong learnability?" (Kearns & Valiant) was answered constructively by AdaBoost (Freund & Schapire, 1997).

## AdaBoost (the historical door in)

Binary labels $y_i\in\{-1,+1\}$, weights $w_i$ initialized uniformly. At round $m$:

1. Fit a weak classifier $f_m$ to the weighted data; let $\varepsilon_m=\sum_{i}w_i\mathbb{1}\{f_m(x_i)\ne y_i\}$ be its weighted error.
2. Set its say $\alpha_m=\frac{1}{2}\log\frac{1-\varepsilon_m}{\varepsilon_m}$.
3. Re-weight: $w_i\leftarrow w_i\, e^{-\alpha_m y_i f_m(x_i)}$ (and normalize) — misclassified points get **up-weighted**.

Predict with $\text{sign}\left(\sum_m\alpha_m f_m(x)\right)$. The training error provably drops exponentially: if every weak learner has edge $\gamma$ ($\varepsilon_m\le\frac{1}{2}-\gamma$), then training error $\le e^{-2\gamma^2 M}$.

The modern understanding (Friedman, Hastie & Tibshirani, 2000): AdaBoost is exactly **stagewise coordinate descent on the exponential loss** $L(y,F)=e^{-yF(x)}$. The re-weighting is not a trick — $w_i \propto e^{-y_iF_{m-1}(x_i)}$ is the derivative of the loss at the current margin. This reframing generalizes to any loss and gives us:

## Gradient Boosting

Think of the ensemble's predictions $\left(F(x_1),\dots,F(x_n)\right)$ as $n$ free parameters and do gradient descent **in function space**. At stage $m$, compute the negative gradient of the loss at the current model — the *pseudo-residuals*:

$$
r_i^{(m)}=-\left.\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}.
$$

We can't move each $F(x_i)$ independently (that would just memorize the training set and generalize nowhere) — so we **project the gradient step onto the class of small trees**: fit a regression tree $f_m$ to the pairs $(x_i, r_i^{(m)})$, then update

$$
F_m(x)=F_{m-1}(x)+\eta\, f_m(x), \qquad 0<\eta\le 1.
$$

Check the special cases: for squared loss, $r_i=y_i-F_{m-1}(x_i)$ — literally the residuals, "fit a tree to the errors of the ensemble so far". For log-loss (classification), $r_i = y_i - p_{m-1}(x_i)$ with $p=\sigma(F)$. Any differentiable loss works — quantile loss, Poisson, ranking losses — which is a large part of gradient boosting's practical power.

## Regularization: the knobs and why they matter

Unlike bagging, **boosting can overfit as $M$ grows** — it keeps reducing bias until it starts fitting noise. The standard controls interact:

* **Learning rate (shrinkage) $\eta$**: smaller steps in function space. Empirically, small $\eta$ (0.01-0.1) with proportionally more trees generalizes better; $\eta$ and $M$ trade off, tune $M$ with early stopping on a validation set.
* **Tree depth**: controls the *interaction order* the model can express — a depth-$k$ tree captures up to $k$-way feature interactions. Depth 3-6 is almost always enough.
* **Subsampling** ("stochastic gradient boosting"): fit each tree on a random fraction of rows (and/or columns). Injects the variance-reduction ideas of bagging into boosting.

## Boosting vs. Bagging in one table

| | Bagging / RF | Boosting |
| --- | --- | --- |
| Trees trained | independently, in parallel | sequentially, each fixing the last |
| Attacks | variance | bias (mainly) |
| Base tree | deep, low-bias | shallow, high-bias |
| More trees | never overfits | overfits — needs early stopping |
| Tuning effort | minimal | real (η, depth, M, subsampling) |
| Typical accuracy on tabular data | good | state of the art |

## Walkthrough

[Walkthrough - Trees & Ensembles in Practice](../walkthroughs/lesson_trees_ensembles.ipynb)
