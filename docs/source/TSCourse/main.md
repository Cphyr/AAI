# Time Series & State-Space Crash Course

A 3-week course that follows the [ML Crash Course](../MLCrashCourse/main.md). This is **not** a
theoretical time-series analysis course. Its goal is the ability to *pose, solve and critically
evaluate* problems that live in the time-series / state-space domain: track classification,
trajectory prediction, filtering, and anomaly detection on tracks.

Read the lesson first, run the walkthrough second, and only then attempt the challenge.

## Three lenses

Every lesson connects to at least one of these, and every challenge is graded against them:

1. **Representation of temporal structure** — lags, state, segments, regimes. What are you feeding the model, and what does that choice throw away?
2. **Evaluation & uncertainty** — what "good" means for *this* task; leakage; calibration.
3. **Assumption–reality mismatch** — nonstationarity, non-Gaussianity, multi-modality, and how to notice when your model's assumptions broke.

![Five framings of one track](../_static/ts/problem_framings.png)

## Suggested 3-Week Schedule

| Week | Days | Material |
| ---- | ---- | -------- |
| 1 | 1 (am) | [Chapter 0 - Problem Formulation & Baselines](chapter_00_problem_formulation/main.md) |
| 1 | 1 (pm)-2 | [Chapter 1 - TS Foundations, Pragmatic](chapter_01_ts_foundations/main.md) |
| 1 | 3-5 | [Chapter 2 - State-Space & Filtering](chapter_02_state_space_filtering/main.md) |
| 2 | 1 | [Chapter 3 - Trajectory Data in Practice](chapter_03_trajectory_data/main.md) |
| 2 | 2-3 | [Chapter 4 - Time-Series & Trajectory Classification](chapter_04_ts_classification/main.md) |
| 2 | 4 (am) | [Chapter 5 - Anomaly Detection on Trajectories](chapter_05_anomaly_detection/main.md) |
| 2-3 | 4 (pm)-5, 1 | [Chapter 6 - Probabilistic Prediction & Uncertainty](chapter_06_probabilistic_prediction/main.md) |
| 3 | 1-3 | [Chapter 7 - Deep Learning for Sequences & Trajectories](chapter_07_deep_learning_sequences/main.md) |
| 3 | 3 | [Chapter 8 - Frontier: Foundation Models & Modern SSMs](chapter_08_frontier_foundation_models/main.md) |
| 3 | 4-5 | [Chapter 9 - Capstone: Critical Reading & Reproduction](chapter_09_capstone/main.md) |

## Stack

`numpy` / `scipy` / `pandas` / `matplotlib` / `scikit-learn` carry every walkthrough — the filters,
DTW, ROCKET-style features and conformal intervals are all implemented from scratch, because the
point is to see the machinery. In a real project reach for `filterpy` (Kalman/IMM),
`aeon`/`sktime` (DTW, ROCKET, catch22), `ruptures` (change points), `statsmodels` (structural TS)
and `pytorch` instead.

```{toctree}
:maxdepth: 2
:hidden:

chapter_00_problem_formulation/main
chapter_01_ts_foundations/main
chapter_02_state_space_filtering/main
chapter_03_trajectory_data/main
chapter_04_ts_classification/main
chapter_05_anomaly_detection/main
chapter_06_probabilistic_prediction/main
chapter_07_deep_learning_sequences/main
chapter_08_frontier_foundation_models/main
chapter_09_capstone/main
```
