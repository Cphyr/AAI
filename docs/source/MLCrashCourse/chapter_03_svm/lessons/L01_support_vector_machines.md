```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - Support Vector Machines

## Motivation

For linearly separable data there are infinitely many separating hyperplanes. Which one should we pick? The SVM's answer: the one **farthest from both classes** — the maximum-margin hyperplane. This single geometric idea unfolds into three of the most important concepts in machine learning: margin-based generalization, convex duality, and the kernel trick. Even in the deep-learning era, SVMs remain excellent for small/medium datasets, and the concepts are everywhere (hinge loss, margins, kernels appear in modern theory constantly).

## Hard-Margin SVM

A hyperplane is $\{x: w^Tx+b=0\}$; the (signed) distance of a point to it is $\frac{w^Tx+b}{\|w\|}$. For labels $y_i\in\{-1,+1\}$, the **margin** is the distance of the closest point. Fixing the scale by $\min_i y_i(w^Tx_i+b)=1$ (the "canonical form"), the margin equals $\frac{1}{\|w\|}$, and maximizing it becomes a convex quadratic program:

$$
\min_{w,b}\ \frac{1}{2}\|w\|^2
\quad\text{s.t.}\quad y_i\left(w^Tx_i+b\right)\ge 1,\ \ i=1,\dots,n.
$$

![https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png](https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png)

Why maximize the margin? Intuition: the fat separator is robust to perturbations of the inputs. Theory: generalization bounds for linear classifiers depend on the margin, not on the dimension — which is what will save us when we go to infinite-dimensional feature spaces below.

## The Dual and the Support Vectors

Introduce Lagrange multipliers $\alpha_i\ge0$ for the constraints. The Lagrangian is
$L=\frac{1}{2}\|w\|^2-\sum_i\alpha_i\left[y_i(w^Tx_i+b)-1\right]$. Setting derivatives to zero: $w=\sum_i\alpha_i y_i x_i$ and $\sum_i\alpha_i y_i=0$. Substituting back gives the **dual problem**:

$$
\max_{\alpha\ge0}\ \sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j\, x_i^Tx_j
\quad\text{s.t.}\quad \sum_i\alpha_i y_i=0.
$$

The KKT **complementary slackness** condition, $\alpha_i\left[y_i(w^Tx_i+b)-1\right]=0$, says $\alpha_i>0$ only for points *on the margin*. These are the **support vectors**: the solution $w=\sum\alpha_iy_ix_i$ is a combination of them alone, and deleting any other training point changes nothing. This sparsity is both an efficiency property and the right mental picture: the SVM summarizes the dataset by its hardest boundary cases.

Note the crucial structural fact: **the dual involves the data only through inner products** $x_i^Tx_j$, and prediction too: $f(x)=\text{sign}\left(\sum_i\alpha_iy_i\,x_i^Tx+b\right)$.

## Soft Margin: Real Data Isn't Separable

Allow violations with slack variables $\xi_i\ge0$:

$$
\min_{w,b,\xi}\ \frac{1}{2}\|w\|^2+C\sum_{i=1}^n\xi_i
\quad\text{s.t.}\quad y_i(w^Tx_i+b)\ge 1-\xi_i,\ \ \xi_i\ge0.
$$

$C$ trades margin width against violations: large $C$ → hard margin, overfitting; small $C$ → wide, tolerant margin. In the dual, the only change is the box constraint $0\le\alpha_i\le C$.

Eliminating $\xi$ gives the unconstrained view, which connects SVMs to everything else you know:

$$
\min_{w,b}\ \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\max\left(0,\ 1-y_i(w^Tx_i+b)\right),
$$

i.e. **$\ell_2$-regularized empirical risk minimization with the hinge loss**. The hinge is a convex surrogate of the 0-1 loss that keeps pushing until a point clears the margin, then stops (unlike log-loss, which never fully stops). Same template as logistic regression — only the loss differs.

## The Kernel Trick

Nonlinear boundaries: map inputs through a feature map $\phi:\mathbb{R}^d\to\mathcal{H}$ and run the (soft-margin) SVM there. Since the dual touches data only via inner products, we never need $\phi$ itself — only the **kernel**

$$
k(x,x')=\langle\phi(x),\phi(x')\rangle.
$$

Replace every $x_i^Tx_j$ with $k(x_i,x_j)$ and you have trained a linear separator in $\mathcal{H}$ — possibly infinite-dimensional — at the cost of computing an $n\times n$ kernel matrix. By Mercer's theorem, any symmetric positive semi-definite function is a valid kernel (it *is* an inner product in some space). The standard menu:

* Linear: $k(x,x')=x^Tx'$ — use when $d\gg n$ (e.g. text), or as a fast baseline.
* Polynomial: $k(x,x')=(x^Tx'+c)^p$ — explicit feature interactions up to order $p$.
* RBF (Gaussian): $k(x,x')=\exp\left(-\gamma\|x-x'\|^2\right)$ — infinite-dimensional, universal approximator; the default nonlinear choice. $\gamma$ sets the length-scale: large $\gamma$ → wiggly boundary (overfit), small $\gamma$ → nearly linear.

Why doesn't an infinite-dimensional feature space instantly overfit? Because generalization is controlled by the **margin**, not the dimension — this is the payoff of the max-margin objective.

## Practical Notes

* **Scale your features.** Kernels are distance-based; one feature in units of thousands dominates everything (unlike trees, SVMs care).
* Tune $C$ and $\gamma$ jointly on a log grid; the good region is usually a diagonal band.
* Training is $O(n^2)$-ish in samples: beyond ~$10^5$ points, use `LinearSVC`/SGD with hinge loss, or approximate kernels with random Fourier features.
* Multi-class is handled by one-vs-rest/one-vs-one reductions; probabilities require an extra calibration step (Platt scaling) — SVMs natively output margins, not probabilities.

## Walkthrough

[Walkthrough - SVMs and Kernels](../walkthroughs/lesson_svm_kernels.ipynb)

## Available Challenges

[Challenge 01 - Kernel Design](../challenges/challenge1_kernel_design.ipynb)
