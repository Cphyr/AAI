```
Author: Cfir Hadar

Tags: Done
```
# Lesson 01 - SVD and PCA

## Motivation

High-dimensional data almost always lives near a lower-dimensional structure: pixels of face images, gene-expression profiles, sensor arrays with correlated channels. Dimensionality reduction serves three distinct purposes — **compression** (keep the signal, drop redundancy), **denoising** (small directions are often noise), and **visualization** (get to 2-3 dimensions). Linear methods (this lesson) are the right tool for the first two; nonlinear methods (next lesson) mostly for the third. Keep this distinction in mind — it is the main practical question of this chapter.

## Singular Value Decomposition

Any matrix $X\in\mathbb{R}^{n\times d}$ (n samples, d features) factorizes as

$$
X=U\Sigma V^T=\sum_{k=1}^{r}\sigma_k u_k v_k^T,
$$

with $U\in\mathbb{R}^{n\times r}$, $V\in\mathbb{R}^{d\times r}$ orthonormal ($U^TU=V^TV=I$), and $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$ the singular values ($r=\text{rank}(X)$). Geometrically: every linear map is a rotation, followed by axis-aligned scaling, followed by another rotation.

The reason SVD matters in data analysis is the **Eckart-Young theorem**: the best rank-$k$ approximation of $X$ — in either Frobenius or spectral norm — is obtained by truncating the SVD,

$$
X_k=\sum_{j=1}^{k}\sigma_j u_j v_j^T
=\arg\min_{\text{rank}(A)\le k}\|X-A\|_F,
\qquad
\|X-X_k\|_F^2=\sum_{j>k}\sigma_j^2.
$$

So the SVD hands you, for every $k$ simultaneously, the optimal $k$-dimensional linear compression *and* tells you exactly how much you lose ($\sum_{j>k}\sigma_j^2$). Everything with a "low-rank" flavor — image compression, latent semantic analysis, matrix-completion recommenders, LoRA fine-tuning of LLMs — is this theorem at work.

## PCA

Principal Component Analysis asks a statistical question: which directions carry the most **variance**? Center the data ($\tilde X = X-\mathbb{1}\bar x^T$) and consider the empirical covariance $S=\frac{1}{n}\tilde X^T\tilde X$. The first principal direction is

$$
w_1=\arg\max_{\|w\|=1} w^TSw,
$$

whose solution (by the Rayleigh-quotient / Lagrange-multiplier argument $Sw=\lambda w$) is the top eigenvector of $S$, with the variance along it equal to $\lambda_1$. Subsequent components maximize variance subject to orthogonality to the previous ones → the top-$k$ eigenvectors.

There is an exactly equivalent second view: PCA finds the $k$-dimensional subspace **minimizing reconstruction error** $\sum_i\|x_i-P x_i\|^2$ (variance maximized = residual minimized, since the two add up to the total variance by Pythagoras).

![https://upload.wikimedia.org/wikipedia/commons/f/f5/GaussianScatterPCA.svg](https://upload.wikimedia.org/wikipedia/commons/f/f5/GaussianScatterPCA.svg)

**PCA = SVD of the centered data matrix.** If $\tilde X=U\Sigma V^T$ then $S=\frac{1}{n}V\Sigma^2V^T$: the principal directions are the right singular vectors $V$, the component variances are $\lambda_j=\sigma_j^2/n$, and the projected coordinates ("scores") are $\tilde XV=U\Sigma$. In practice PCA is always *computed* via SVD (never form $S$ explicitly — squaring the matrix squares its condition number). The practical differences to remember: PCA is SVD **after centering** (and usually standardizing), and SVD is more general (applies to any matrix, no probabilistic interpretation needed).

Choosing $k$: plot the explained-variance ratio $\frac{\sum_{j\le k}\lambda_j}{\sum_j\lambda_j}$ and look for the elbow, or keep e.g. 95% of variance. For visualization $k=2$; for compression/denoising, the spectrum decides.

## What Linear Methods Can and Cannot Do

* **Do**: optimal linear compression; decorrelate features; denoise (small-$\sigma$ directions are dominated by noise when signal is low-rank — truncation is a denoiser); speed up and regularize downstream models; whitening.
* **Cannot**: capture curved (nonlinear) structure. The classic counterexample is the "Swiss roll": a 2D sheet rolled up in 3D. PCA sees large variance in all three coordinates and happily projects the roll onto itself, destroying the neighborhood structure. Data on a nonlinear manifold needs the next lesson's tools.

Also remember: **PCA is unsupervised** — the highest-variance direction is not necessarily the most discriminative one for your labels (that would be LDA). And PCA is sensitive to feature scale: standardize first unless the units are already comparable.

## Walkthrough

[Walkthrough - Comparing Dimensionality Reduction Methods](../walkthroughs/lesson_dimensionality_reduction.ipynb)
