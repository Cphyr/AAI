```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - State-Space Models as Deep Architectures

## Motivation

The most interesting recent sequence architectures are, structurally, the state-space models of
Chapter 2 with the matrices *learned* instead of derived from physics. Reading S4 and Mamba as
"Kalman without the Kalman" is not a cute analogy — it tells you exactly what they buy and what
they give up.

## The same two equations

A continuous linear SSM,

$$
\dot h(t)=A\,h(t)+B\,u(t),\qquad y(t)=C\,h(t)+D\,u(t),
$$

discretised with step $\Delta$ (zero-order hold: $\bar A=e^{A\Delta}$,
$\bar B=A^{-1}(e^{A\Delta}-I)B$), becomes

$$
h_t=\bar A h_{t-1}+\bar B u_t,\qquad y_t=Ch_t+Du_t .
$$

Compare Chapter 2 Lesson 01: identical structure. Two differences, both decisive.

1. $A,B,C,D$ are **learned parameters**, not physics, and there is no process/measurement noise —
   the model is deterministic, so there is no posterior and no uncertainty.
2. Because the recursion is *linear*, it can be unrolled into a convolution:
   $y=\bar K*u$ with kernel $\bar K=(C\bar B,\ C\bar A\bar B,\ C\bar A^2\bar B,\dots)$, computable
   in $O(L\log L)$ by FFT. So it trains in parallel like a CNN and runs recurrently at inference in
   $O(1)$ per step and $O(1)$ memory — the property no transformer has.

## S4: making the long-range memory work

A randomly initialised $\bar A$ either forgets immediately or explodes. **S4** (Gu et al., 2021)
initialises $A$ with the **HiPPO** matrix, derived to make the state an optimal online compression
of the input history onto a basis of orthogonal polynomials — the state is, literally, a running
summary of the whole past. That single choice is what makes long-range memory work; ablate it and
performance collapses to ordinary RNN levels.

The remaining machinery is computational: a diagonal-plus-low-rank structure for $A$ so the
convolution kernel can be computed efficiently. **S4D** shows a purely diagonal parameterisation
with the right initialisation works nearly as well and is far simpler — the version to read first.

S4 posted the first strong results on Long Range Arena (16k-step sequences) and on raw audio,
territory where transformers are impractical and RNNs fail.

## Mamba: selectivity

S4's $\bar A,\bar B,\bar C$ are the same at every time step — the model is linear time-invariant,
which means it cannot decide to *ignore* an input based on content. **Mamba** (Gu & Dao, 2023) makes
$B$, $C$ and $\Delta$ functions of the current input ("selective SSM"). Content-dependent $\Delta$
is the key: a large $\Delta$ means "take a big step, forget the past", a small one means "hold
state". That is a learned gate, and it recovers the ability to do the associative recall tasks S4
fails at.

The cost is that time-varying parameters destroy the convolution trick. Mamba replaces it with a
hardware-aware **parallel scan** (associative scan) kept in fast memory, giving linear-time
training and constant-memory inference. Mamba-2 later reframes the whole thing as a structured
matrix ("state-space duality"), connecting it to linear attention.

The practical claim: transformer-quality modeling at $O(L)$ instead of $O(L^2)$, with a fixed-size
state at inference. The practical caveat: on tasks needing exact recall of arbitrary earlier
tokens, a fixed-size state is a real bottleneck — hybrid architectures (a few attention layers
among many SSM layers) currently win.

## Bridge back to Chapter 2

| | Classical SSM (Kalman) | Deep SSM (S4/Mamba) |
| --- | --- | --- |
| $A,B,C$ | from physics/domain | learned |
| Noise | explicit $Q,R$; posterior covariance | none; deterministic |
| Output | distribution over state | features |
| Data needed | none (just tuning) | large |
| Interpretability | state has units and meaning | state is a latent code |
| Failure diagnosis | NIS/NEES, innovations | loss curves, ablations |
| Inference cost | $O(1)$/step | $O(1)$/step (recurrent mode) |

What deep SSMs buy: long-range memory without quadratic cost, no feature engineering, and one
architecture across modalities. What they give up: uncertainty quantification, physical grounding,
data efficiency, and the ability to say *why* an estimate moved.

The two are not rivals; the interesting territory is between them — learned dynamics with an
explicit noise model, differentiable Kalman filters (backprop through the filter to learn $F,Q,R$
from data), deep state-space models with variational inference (Deep Kalman Filters, DVBF), or
neural networks that output the *parameters* of a classical filter for a physics-constrained
estimate. That is your research hook.

## Research hook (bring to the capstone)

Write half a page on 1-2 concrete transfers, in either direction. Some starting points:

* Give a Mamba-style selective SSM an explicit process-noise term, and read $\Delta$ as an
  adaptive-$Q$ mechanism (Ch.2 L03). Does IMM-style mode-mixing have a deep analogue?
* Use HiPPO's optimal-history-compression idea as a *feature extractor* for trajectory
  classification (Ch.4) — a principled fixed representation to compare against ROCKET.
* Learn $Q$ and $R$ end-to-end through a differentiable Kalman filter on real tracks, and compare
  its NIS consistency against a hand-tuned filter.
* Take the association problem (Ch.2 L04) and ask what a learned model would have to represent to
  defer decisions the way MHT does.

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| Fixed-size state suffices | exact recall of arbitrary past tokens | fails copy/retrieval tasks | hybrid attention+SSM |
| Linear recurrence is enough | strongly nonlinear dynamics | underfits | nonlinearities between SSM layers; selectivity |
| Initialisation is a detail | random $A$ instead of HiPPO/S4D | long-range performance collapses | use the prescribed init |
| Deterministic state is fine | you need uncertainty | over-confident downstream decisions | keep a classical filter, or go variational |
| Long-range benchmarks transfer | your sequences are short | no gain over a TCN or ridge | Ch.0 baselines, again |

**Lens check:** lens 1 (state as learned representation) and lens 3 (what the model *cannot*
represent — uncertainty — is exactly what Chapter 2 was built to give you).
