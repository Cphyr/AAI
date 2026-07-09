```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Nonlinear Methods: t-SNE, UMAP and Friends

## Motivation

PCA preserves *global*, *linear* structure: large pairwise distances and variance directions. But data on a curved manifold (the Swiss roll, image manifolds, single-cell RNA data) is better described by its *local neighborhoods* — who is close to whom. Nonlinear methods trade the global fidelity of PCA for local-neighborhood fidelity. The central message of this lesson: **each method preserves a different property of the data, and that property determines what tasks it is suitable for.** There is no "best" method, only a best match between preserved property and your question.

## Kernel PCA (the bridge)

Map the data through a feature map $\phi$ and do PCA in that space. By the kernel trick you never compute $\phi$ — only the kernel matrix $K_{ij}=k(x_i,x_j)$, whose (centered) eigendecomposition gives the components. With an RBF kernel, kernel PCA can unfold curved structure that linear PCA cannot. Its weaknesses: $O(n^2)$ memory, the result depends strongly on kernel choice, and there is no natural inverse map. In practice it has mostly been displaced by the graph-based methods below, but it is the cleanest conceptual bridge from linear to nonlinear.

## t-SNE

t-SNE (van der Maaten & Hinton, 2008) is explicitly a **neighborhood-preserving visualization** method. It converts distances to conditional probabilities in the high-dimensional space:

$$
p_{j|i}=\frac{\exp\left(-\|x_i-x_j\|^2/2\sigma_i^2\right)}{\sum_{k\ne i}\exp\left(-\|x_i-x_k\|^2/2\sigma_i^2\right)},
\qquad p_{ij}=\frac{p_{j|i}+p_{i|j}}{2n},
$$

where each $\sigma_i$ is tuned per-point so that the effective number of neighbors equals a user-set **perplexity** ($\text{Perp}(p_{\cdot|i}) = 2^{H(p_{\cdot|i})}$, typical values 5-50). In the low-dimensional embedding $\{y_i\}$ it uses a **Student-t** (Cauchy) kernel:

$$
q_{ij}=\frac{\left(1+\|y_i-y_j\|^2\right)^{-1}}{\sum_{k\ne l}\left(1+\|y_k-y_l\|^2\right)^{-1}},
$$

and minimizes the KL divergence $\text{KL}(P\|Q)=\sum_{ij}p_{ij}\log\frac{p_{ij}}{q_{ij}}$ by gradient descent.

Two design choices carry all the intuition:

1. **KL is asymmetric**: a large $p_{ij}$ paired with a small $q_{ij}$ (true neighbors placed far apart) costs a lot; a small $p_{ij}$ with a large $q_{ij}$ costs little. So t-SNE fights hard to keep neighbors together but doesn't much care where it puts non-neighbors → local structure is faithful, **global arrangement is nearly meaningless**.
2. **Heavy tails in the embedding**: moderate high-D distances can map to large 2D distances without penalty. This solves the *crowding problem* (there is exponentially less room in 2D than in high-D at a given radius) and is what makes t-SNE plots form well-separated clusters.

Consequences you must internalize before reading a t-SNE plot: cluster *sizes* and *inter-cluster distances* in the embedding mean nothing; different perplexities give different pictures; the algorithm can show "clusters" in pure noise; the map is non-parametric (no out-of-sample transform without re-running) and $O(n^2)$ naively (Barnes-Hut brings it to $O(n\log n)$).

## UMAP

UMAP (McInnes et al., 2018) is t-SNE's practical successor. Same skeleton — build a weighted k-NN graph in high dimensions, then lay it out in low dimensions by minimizing a divergence (a cross-entropy over edge probabilities, optimized with negative sampling). The differences that matter:

* **Much faster** and scales to millions of points; supports out-of-sample `transform()` for new data and can run supervised.
* The `n_neighbors` parameter (analog of perplexity) sets the local/global tradeoff explicitly; `min_dist` controls how tightly points pack.
* Preserves somewhat more global structure than t-SNE in practice, though inter-cluster distances are still not to be trusted.

A fair one-line summary: **use UMAP where you would use t-SNE, unless you need the specific t-SNE literature-comparability**; treat both as visualization tools, not as general-purpose feature extractors.

## Choosing a Method (the actual point of this lesson)

| Method | Preserves | Out-of-sample | Scales | Use for |
| ------ | --------- | ------------- | ------ | ------- |
| PCA/SVD | global variance, large distances | yes (linear map) | excellent | compression, denoising, decorrelation, preprocessing for models |
| Kernel PCA | nonlinear structure via chosen kernel | approximately | $O(n^2)$ | small data with known kernel structure |
| t-SNE | local neighborhoods only | no | $O(n\log n)$ | 2D visualization, cluster inspection |
| UMAP | local neighborhoods (+ some global) | yes | very good | visualization at scale, k-NN-graph features |

Rules of thumb:

* Feeding features to a downstream model → PCA (or just don't reduce; trees don't care about dimension much).
* "Show me the structure of my dataset" → UMAP/t-SNE, **but run PCA first to ~50 dimensions** (standard practice: denoises and speeds up the k-NN graph).
* Never cluster on a 2D t-SNE/UMAP embedding and report the result as ground truth — cluster in the original (or PCA) space and use the embedding to *look*.
* Always check stability: rerun with a different seed/perplexity; structure that survives is real, structure that doesn't is an artifact.

## Walkthrough

[Walkthrough - Comparing Dimensionality Reduction Methods](../walkthroughs/lesson_dimensionality_reduction.ipynb)

## Available Challenges

[Challenge 01 - Embed and Explain](../challenges/challenge1_embed_and_explain.ipynb)
