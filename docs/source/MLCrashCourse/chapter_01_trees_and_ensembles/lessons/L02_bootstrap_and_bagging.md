```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - The Bootstrap and Bagging

## Motivation

We ended the last lesson with a diagnosis: trees are low-bias, high-variance. The cure is averaging — but averaging requires many datasets, and we only have one. The **bootstrap** is the statistical trick that manufactures "many datasets" out of one, and **bagging** is its application to prediction. The bootstrap is worth understanding on its own: it is one of the most useful tools in applied statistics (confidence intervals for *any* statistic, with no closed-form math).

## The Bootstrap (the statistics)

Setup: we observe $X_1,\dots,X_n \stackrel{iid}{\sim} F$ and compute a statistic $\hat\theta=s(X_1,\dots,X_n)$ (a mean, a median, a model's AUC...). We want the sampling distribution of $\hat\theta$ — e.g., its standard error — but we cannot draw fresh samples from $F$.

**The plug-in principle**: replace the unknown $F$ with the *empirical distribution* $\hat F_n$, which puts mass $\frac{1}{n}$ on each observed point. Sampling from $\hat F_n$ is exactly *sampling $n$ points from our data with replacement*. So:

1. For $b=1,\dots,B$: draw $X_1^{*b},\dots,X_n^{*b}$ with replacement from the data; compute $\hat\theta^{*b}=s(X^{*b})$.
2. Estimate the standard error by the empirical spread of the replicates:

$$
\widehat{se}_{boot}=\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(\hat\theta^{*b}-\bar\theta^*\right)^2},
\qquad \bar\theta^*=\frac{1}{B}\sum_b\hat\theta^{*b}.
$$

The approximation being made is $\text{Var}_{\hat F_n}(\hat\theta) \approx \text{Var}_F(\hat\theta)$, which is justified because $\hat F_n \to F$ (Glivenko–Cantelli). The percentile interval $[\hat\theta^*_{(\alpha/2)},\hat\theta^*_{(1-\alpha/2)}]$ gives a confidence interval for free. This works for statistics where no formula exists — try deriving the standard error of a median analytically and you'll appreciate it.

**The 63.2% fact.** The probability that a given sample appears in a bootstrap resample is

$$
P(i \in X^{*b}) = 1-\left(1-\frac{1}{n}\right)^n \xrightarrow{n\to\infty} 1-e^{-1}\approx 0.632.
$$

So each resample contains about 63% of the distinct points; the remaining ~37% are "out-of-bag" (OOB). Hold this thought.

## Bagging = Bootstrap AGGregatING

Apply the bootstrap to models (Breiman, 1996): train a tree $\hat f^{*b}$ on each bootstrap resample and average,

$$
\hat f_{bag}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat f^{*b}(x)
$$

(majority vote for classification). Why does this help? Suppose the individual predictors have variance $\sigma^2$ and pairwise correlation $\rho$. Then

$$
\text{Var}\left(\frac{1}{B}\sum_b \hat f^{*b}(x)\right)
=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2.
$$

Derivation: the variance of a mean of $B$ identically distributed variables has $B$ diagonal terms $\sigma^2$ and $B(B-1)$ covariance terms $\rho\sigma^2$; divide by $B^2$. Two consequences:

1. The second term vanishes as $B\to\infty$: **more trees never hurt** (unlike boosting, bagging does not overfit in $B$; you just pay compute).
2. The first term $\rho\sigma^2$ does **not** vanish. Bootstrap resamples share ~63% of their points, so the trees are substantially correlated, and $\rho$ is the floor on the achievable variance. Reducing $\rho$ is exactly the extra idea of random forests (next lesson).

Note what bagging does *not* do: the bias of $\hat f_{bag}$ equals the bias of a single (bootstrap-trained) tree. Averaging only attacks variance — which is why we bag deep, unpruned, low-bias trees, and never bag linear models (a bagged linear model is essentially still a linear model: the average of linear functions is linear).

## Out-of-Bag Evaluation

Each tree never saw its ~37% OOB samples, so for every training point there are ≈$0.37B$ trees for which it is a legitimate test point. Averaging those trees' predictions gives the **OOB error** — an honest, cross-validation-quality estimate of generalization error, with zero extra computation. In practice: when using bagging/random forests, you often don't need a separate validation split at all.

## Summary

* Bootstrap = sample from $\hat F_n$ (i.e., resample your data with replacement) to approximate the sampling distribution of any statistic.
* Bagging = train on $B$ bootstrap resamples and average; variance $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$, bias unchanged.
* Each resample leaves out ~37% of points → free OOB generalization estimate.

## Walkthrough

[Walkthrough - Trees & Ensembles in Practice](../walkthroughs/lesson_trees_ensembles.ipynb) (includes a bootstrap confidence-interval demo).
