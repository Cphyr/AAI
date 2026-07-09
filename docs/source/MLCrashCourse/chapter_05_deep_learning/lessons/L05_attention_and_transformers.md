```
Author: Cfir Hadar

Tags: Done
```
# Lesson 05 - Attention and Transformers

## Motivation

RNNs (Data Science 101, Lesson 04) process a sequence step by step: information from position 1 must survive $t$ recurrence steps to influence position $t$. This creates two problems — long-range dependencies fade (even with LSTM gating), and computation is inherently *sequential* (no parallelism across time). Attention solves both at once: let every position look **directly** at every other position, in a single parallelizable operation. The transformer (Vaswani et al., 2017, "Attention Is All You Need") is the architecture built purely from this primitive, and it now dominates language, vision, and increasingly time series.

## Scaled Dot-Product Attention

The mental model is a **soft dictionary lookup**. Each position emits a *query* ("what am I looking for?"), a *key* ("what do I contain?"), and a *value* ("what do I offer?"). For a sequence $X\in\mathbb{R}^{n\times d}$:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V .
$$

Read it row by row: position $i$'s output is a weighted average of all value vectors, with weights $\text{softmax}_j\left(q_i^Tk_j/\sqrt{d_k}\right)$ — how well $i$'s query matches each position's key. The $\sqrt{d_k}$ matters: for random vectors with unit-variance entries, $q^Tk$ has variance $d_k$; unscaled, the softmax saturates (one weight ≈ 1, rest ≈ 0) and its gradients vanish. Dividing by $\sqrt{d_k}$ keeps logits at $O(1)$.

Note what changed relative to an RNN: any position reaches any other in **one step** (path length 1 instead of $t$), all positions are computed in parallel as a couple of matrix multiplications — at the cost of $O(n^2)$ time and memory in sequence length. That quadratic cost is the transformer's main tax (and the motivation for a whole zoo of efficient-attention variants).

## Multi-Head Attention

One attention pattern per layer is too little: you may simultaneously want "attend to the previous token", "attend to the subject of the sentence", "attend to similar values". Run $h$ attention operations in parallel on lower-dimensional projections ($d_k=d/h$) and concatenate:

$$
\text{MultiHead}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O .
$$

Same total compute as one full-width head, but $h$ independent attention patterns.

## The Transformer Block

A transformer is a stack of identical blocks, each combining the ingredients you already know from CNN lessons (residual connections, normalization):

$$
\begin{aligned}
Z &= X+\text{MultiHead}(\text{LN}(X)) \\
X' &= Z+\text{MLP}(\text{LN}(Z))
\end{aligned}
$$

where the MLP is position-wise (two linear layers with a nonlinearity, applied to each position independently — this is where most of the parameters live), LN is layer normalization, and the residual connections make 100-layer stacks trainable.

<img src="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png" style="width: 45%">

**Positional information.** Attention is permutation-equivariant — shuffle the sequence and outputs shuffle accordingly. Order must be injected explicitly. The original design adds sinusoidal encodings $PE_{(pos,2i)}=\sin\left(pos/10000^{2i/d}\right)$, $PE_{(pos,2i+1)}=\cos(\cdot)$; learned position embeddings, and more recently *relative/rotary* encodings (RoPE), are common. For time series, positional encodings can carry timestamps and calendar features — a natural fit.

**Masking and the three standard configurations:**

* **Encoder** (BERT-style): full bidirectional attention; use for understanding/embedding tasks.
* **Decoder** (GPT-style): a *causal mask* sets attention logits to $-\infty$ for $j>i$, so position $i$ sees only the past; trained by next-token prediction; use for generation and forecasting.
* **Encoder-decoder** (original, T5-style): the decoder additionally *cross-attends* to encoder outputs ($Q$ from decoder, $K,V$ from encoder); use for sequence-to-sequence (translation, and seq2seq forecasting).

## Why Transformers Won

* **Parallel training**: no recurrence → full GPU utilization; scale to billions of parameters.
* **Direct long-range interactions**: constant path length between any two positions.
* **Scaling laws**: loss improves predictably with model/data/compute — transformers keep converting scale into quality where RNNs saturated.
* Weak inductive bias (unlike CNN locality): a drawback on small data (they need more data or pretraining) and the reason pretrained models + fine-tuning is the standard workflow.

For this course's purposes, transformers reappear in [Chapter 6](../../chapter_06_time_series/lessons/L03_deep_learning_for_time_series.md) as forecasting backbones.

## Walkthrough

[Walkthrough - Attention and a Tiny Transformer](../walkthroughs/lesson_attention_transformer.ipynb) — implement scaled dot-product attention from scratch, visualize attention maps, and train a tiny causal transformer on a toy sequence task.
