```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - K-Means and Mixture Models

## Motivation

Clustering is unsupervised: find groups in unlabeled data. The uncomfortable truth to state up front is that **"cluster" has no single definition** — compact blobs, dense regions, connected shapes, and hierarchy are *different* formalizations, and each algorithm optimizes exactly one of them. That is why this chapter is organized around conceptual differences: choosing a clustering algorithm *is* choosing a definition of "cluster". This lesson covers the *centroid/model-based* family; the next covers *density-based* and *hierarchical* methods.

## K-Means

Definition of cluster: **a compact blob around a center**. Given $k$, minimize within-cluster squared distances:

$$
\min_{C_1,\dots,C_k,\ \mu_1,\dots,\mu_k}\ \sum_{j=1}^{k}\sum_{i\in C_j}\|x_i-\mu_j\|^2.
$$

Exact minimization is NP-hard; **Lloyd's algorithm** alternates two steps, each of which cannot increase the objective:

1. **Assign**: $C_j\leftarrow\{i:\ \|x_i-\mu_j\|\le\|x_i-\mu_l\|\ \forall l\}$ (each point to its nearest center).
2. **Update**: $\mu_j\leftarrow\frac{1}{|C_j|}\sum_{i\in C_j}x_i$ (the mean minimizes summed squared distance).

The objective is monotonically non-increasing and there are finitely many partitions, so it converges — but only to a *local* minimum that depends on initialization.

![https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif)

**k-means++ initialization** fixes most of the initialization pain: pick the first center uniformly, then pick each next center with probability proportional to $D(x)^2$, the squared distance to the nearest already-chosen center. This spreads the seeds and guarantees $\mathbb{E}[\text{cost}]\le 8(\ln k+2)\cdot\text{OPT}$. It is the default in sklearn; also run `n_init` restarts and keep the best.

**Built-in assumptions** (each one is a failure mode you will see in the walkthrough):

* Squared Euclidean objective → clusters are implicitly **spherical, equal-variance, similar-size**; elongated or differently-scaled clusters get cut wrongly.
* Every point is assigned → **no notion of noise/outliers**, and outliers drag means.
* $k$ is an input — the algorithm never says "there are actually 3 groups".
* Feature scaling matters (distance-based).

Choosing $k$: the objective decreases monotonically in $k$, so you cannot pick $k$ by the objective alone. Use the **elbow** of the cost curve, or better, the **silhouette score**: for each point, with $a$ = mean distance to its own cluster and $b$ = mean distance to the nearest other cluster, $s=\frac{b-a}{\max(a,b)}\in[-1,1]$; average over points and pick $k$ maximizing it.

## Gaussian Mixture Models: K-Means Grown Up

Replace "blob" with an explicit probabilistic model: data is drawn from a mixture

$$
p(x)=\sum_{j=1}^{k}\pi_j\,\mathcal{N}\left(x\mid\mu_j,\Sigma_j\right),
\qquad \sum_j\pi_j=1 .
$$

Fitting by maximum likelihood has no closed form (the log of a sum), so we use **Expectation-Maximization**, which looks exactly like a soft Lloyd's algorithm:

* **E-step** — soft assignment via Bayes' rule (the *responsibilities*):

$$
  \gamma_{ij}=\frac{\pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}{\sum_{l}\pi_l\,\mathcal{N}(x_i\mid\mu_l,\Sigma_l)} .
$$

* **M-step** — weighted maximum-likelihood updates:

$$
  \mu_j=\frac{\sum_i\gamma_{ij}x_i}{\sum_i\gamma_{ij}},\qquad
  \Sigma_j=\frac{\sum_i\gamma_{ij}(x_i-\mu_j)(x_i-\mu_j)^T}{\sum_i\gamma_{ij}},\qquad
  \pi_j=\frac{1}{n}\sum_i\gamma_{ij} .
$$

Each iteration provably increases the likelihood (EM maximizes a lower bound that is tight at the current parameters). In the limit of shared covariances $\Sigma_j=\sigma^2 I$ with $\sigma\to0$, the responsibilities harden to nearest-center assignment and EM *becomes* k-means — which tells you precisely which assumptions k-means silently makes.

What GMMs buy over k-means: **elliptical clusters** (full covariances), **soft memberships** with calibrated uncertainty, per-cluster sizes/shapes, a likelihood — so $k$ can be chosen with information criteria ($\text{BIC}=-2\log\hat L+p\log n$, pick the minimum), and a generative model you can sample from. The costs: more parameters (full $\Sigma_j$ is $O(d^2)$ each — regularize or constrain on small data), still local optima, still assumes "cluster = one Gaussian-ish blob".

## What This Family Cannot Do

Both k-means and GMM define clusters as *convex-ish blobs around a center*. Two half-moons, concentric rings, or clusters of wildly different densities defeat them by construction — no amount of tuning helps, because the *definition* of cluster is wrong for that data. For those shapes you need density- or connectivity-based definitions: the next lesson.

## Walkthrough

[Walkthrough - Clustering Methods Compared](../walkthroughs/lesson_clustering_comparison.ipynb)
