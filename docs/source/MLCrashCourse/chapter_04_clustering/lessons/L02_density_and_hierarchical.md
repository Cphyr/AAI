```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Density-Based and Hierarchical Clustering

## Motivation

The previous lesson's methods define a cluster as a blob around a center. This lesson covers two alternative definitions: **a cluster is a connected region of high density** (DBSCAN), and **clusters are nested structure at every scale** (hierarchical clustering). Both remove the two most painful requirements of k-means/GMM: knowing $k$ in advance, and convex cluster shapes.

## DBSCAN

DBSCAN (Ester et al., 1996) has two parameters: a radius $\varepsilon$ and a count `minPts`. Definitions:

* A point is a **core point** if at least `minPts` points lie within distance $\varepsilon$ of it (its $\varepsilon$-neighborhood is dense).
* A point is **directly density-reachable** from a core point if it lies in that core point's $\varepsilon$-neighborhood.
* A **cluster** is a maximal set of points connected through chains of core points; non-core points in a cluster are **border points**.
* Points reachable from no core point are **noise** — DBSCAN is the rare algorithm with an explicit outlier output.

![https://upload.wikimedia.org/wikipedia/commons/a/af/DBSCAN-Illustration.svg](https://upload.wikimedia.org/wikipedia/commons/a/af/DBSCAN-Illustration.svg)

*(Here minPts = 4: red are core points, yellow are border points reachable from cores, blue N is noise.)*

The consequences of this definition, point by point against k-means:

* **Arbitrary shapes**: clusters are connected dense regions — half-moons, rings, and snakes are fine, since connectivity is chained locally.
* **$k$ is discovered**, not given: the number of connected dense regions is an *output*.
* **Noise handling**: sparse points are labeled $-1$, not forced into a cluster.
* **The catch — one global density threshold**: $\varepsilon$/`minPts` define a single density cutoff for the whole dataset. Clusters of *different densities* cannot all be captured: an $\varepsilon$ tight enough for the dense cluster shatters the sparse one into noise. This is DBSCAN's main practical failure mode.

Parameter selection: set `minPts` ≈ $2d$ (or just 5-10), then plot each point's distance to its `minPts`-th neighbor, sorted (the *k-distance plot*), and put $\varepsilon$ at the elbow. Complexity is $O(n\log n)$ with a spatial index.

Two successors worth knowing by name: **OPTICS** (orders points by reachability, effectively sweeping all $\varepsilon$ at once) and **HDBSCAN** (a hierarchy over density levels + stable-cluster extraction), both addressing the variable-density weakness — HDBSCAN is an excellent modern default when you'd otherwise reach for DBSCAN.

## Hierarchical Clustering

Instead of one partition, build a **tree of nested clusterings** — a *dendrogram*. The standard approach is **agglomerative** (bottom-up): start with $n$ singleton clusters and repeatedly merge the two closest clusters, where "closest" is defined by the **linkage**:

$$
\begin{aligned}
\text{single:}\quad & d(A,B)=\min_{a\in A,b\in B}\|a-b\| \\
\text{complete:}\quad & d(A,B)=\max_{a\in A,b\in B}\|a-b\| \\
\text{average:}\quad & d(A,B)=\frac{1}{|A||B|}\sum_{a\in A,b\in B}\|a-b\| \\
\text{Ward:}\quad & d(A,B)=\frac{|A||B|}{|A|+|B|}\,\|\mu_A-\mu_B\|^2
\end{aligned}
$$

The linkage *is* the cluster definition, and each has a distinct personality:

* **Single linkage** = connectivity: follows chains, finds elongated/non-convex shapes (it computes the minimum spanning tree), but suffers **chaining** — two blobs joined by a thin bridge of points merge early.
* **Complete linkage** = compactness: every pair in the merged cluster must be close → compact, roughly equal-diameter clusters; sensitive to outliers (a single far point blocks a merge).
* **Average linkage**: the compromise; a solid default.
* **Ward** minimizes the *increase in within-cluster variance* at each merge — the hierarchical sibling of k-means (same objective, greedy nested version); prefers spherical, similar-size clusters.

![https://upload.wikimedia.org/wikipedia/commons/a/ad/Hierarchical_clustering_simple_diagram.svg](https://upload.wikimedia.org/wikipedia/commons/a/ad/Hierarchical_clustering_simple_diagram.svg)

Reading a dendrogram: the height of each merge is the linkage distance at which it happened; cutting the tree at height $h$ yields a flat clustering, and a **long vertical gap** between merges signals a natural number of clusters. The hierarchy itself is often the deliverable — taxonomies, phylogenetics, organizing sensors/assets by similarity — a structure flat methods simply do not produce.

Costs: naive agglomerative clustering is $O(n^3)$ time / $O(n^2)$ memory ($O(n^2)$ time for single linkage via MST) — fine up to ~$10^4$-$10^5$ points, not beyond. No noise concept; merges are greedy and never undone.

## Choosing an Algorithm (summary of the chapter)

| | k-means | GMM | DBSCAN/HDBSCAN | Hierarchical |
| --- | --- | --- | --- | --- |
| Cluster = | blob around center | Gaussian component | connected dense region | nested tree (linkage-defined) |
| $k$ | input | input (BIC helps) | output | choose by cutting the tree |
| Shapes | spherical | elliptical | arbitrary | depends on linkage |
| Noise/outliers | no | soft | explicit | no |
| Scales to | huge ($O(nkd)$/iter) | large | large ($O(n\log n)$) | small/medium |
| Output | partition | probabilities | partition + noise | dendrogram |

Validation without labels is genuinely hard: internal metrics (silhouette, Davies-Bouldin) reward *compact, well-separated* clusters — i.e., they share k-means' bias, and will underrate a correct DBSCAN result on rings. Use them within a family, not across families; when possible, validate by stability (subsample and re-cluster: do the clusters persist?) and — above all — by whether the clusters *mean* something downstream.

## Walkthrough

[Walkthrough - Clustering Methods Compared](../walkthroughs/lesson_clustering_comparison.ipynb)

## Available Challenges

[Challenge 01 - Cluster the Unknown](../challenges/challenge1_cluster_the_unknown.ipynb)
