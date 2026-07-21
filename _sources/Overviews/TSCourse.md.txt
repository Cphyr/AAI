# Time Series & State-Space Crash Course

A 3-week course following the [ML Crash Course](MLCrashCourse.md). See the
[course page](../TSCourse/main.md) for the schedule and the three recurring lenses.

## Chapters:

### (0) Problem Formulation & Baselines

```
Tag: done

Author: Cfir Hadar
```

* [Problem Types & Objectives](../TSCourse/chapter_00_problem_formulation/lessons/L01_problem_types.md)
* [Baselines & Sanity Checks](../TSCourse/chapter_00_problem_formulation/lessons/L02_baselines.md)

### (1) TS Foundations, Pragmatic

```
Tag: done

Author: Cfir Hadar
```

* [Stationarity as a Modeling Convenience](../TSCourse/chapter_01_ts_foundations/lessons/L01_stationarity.md)
* [ACF/PACF & the Spectral View as Diagnostics](../TSCourse/chapter_01_ts_foundations/lessons/L02_acf_and_spectrum.md)
* [Evaluation for Temporal Data](../TSCourse/chapter_01_ts_foundations/lessons/L03_temporal_evaluation.md)

[Challenge - Trick-Series Diagnosis](../TSCourse/chapter_01_ts_foundations/challenges/challenge1_trick_series.ipynb)

### (2) State-Space & Filtering

```
Tag: done

Author: Cfir Hadar
```

* [Linear-Gaussian SSMs & the Kalman Filter](../TSCourse/chapter_02_state_space_filtering/lessons/L01_kalman_filter.md)
* [Beyond Linear-Gaussian: EKF, UKF, Particle Filters](../TSCourse/chapter_02_state_space_filtering/lessons/L02_nonlinear_filters.md)
* [Motion Models & Maneuvering Targets](../TSCourse/chapter_02_state_space_filtering/lessons/L03_motion_models_imm.md)
* [Multi-Target Essentials](../TSCourse/chapter_02_state_space_filtering/lessons/L04_multi_target.md)

[Walkthrough](../TSCourse/chapter_02_state_space_filtering/walkthroughs/lesson_maneuvering_target.ipynb) |
[Challenge - Sensor Fusion with Non-Gaussian Noise](../TSCourse/chapter_02_state_space_filtering/challenges/challenge1_nongaussian_fusion.ipynb)

### (3) Trajectory Data in Practice

```
Tag: done

Author: Cfir Hadar
```

* [Irregular Sampling, Dropouts, Resampling](../TSCourse/chapter_03_trajectory_data/lessons/L01_irregular_sampling.md)
* [Coordinate Frames & Kinematic Features](../TSCourse/chapter_03_trajectory_data/lessons/L02_frames_and_features.md)
* [Segmentation of Long Recordings](../TSCourse/chapter_03_trajectory_data/lessons/L03_segmentation.md)

[Walkthrough](../TSCourse/chapter_03_trajectory_data/walkthroughs/lesson_trajectory_pipeline.ipynb)

### (4) Time-Series & Trajectory Classification

```
Tag: done

Author: Cfir Hadar
```

* [Distance-Based & Feature-Based Classification](../TSCourse/chapter_04_ts_classification/lessons/L01_distance_and_features.md)
* [Random-Feature & Deep Classifiers](../TSCourse/chapter_04_ts_classification/lessons/L02_rocket_and_deep.md)

[Challenge - Classify Trajectory Types](../TSCourse/chapter_04_ts_classification/challenges/challenge1_classify_trajectories.ipynb)

### (5) Anomaly Detection on Trajectories

```
Tag: done

Author: Cfir Hadar
```

* [Framings & Evaluation Under Label Scarcity](../TSCourse/chapter_05_anomaly_detection/lessons/L01_anomaly_framings.md)

[Challenge - Flag Anomalous Tracks in Unlabeled Data](../TSCourse/chapter_05_anomaly_detection/challenges/challenge1_unlabeled_anomalies.ipynb)

### (6) Probabilistic Prediction & Uncertainty

```
Tag: done

Author: Cfir Hadar
```

* [From Point Forecasts to Distributions](../TSCourse/chapter_06_probabilistic_prediction/lessons/L01_quantiles_and_pinball.md)
* [Conformal Prediction for Time Series](../TSCourse/chapter_06_probabilistic_prediction/lessons/L02_conformal.md)
* [Bayesian Structural TS, CRPS & Calibration](../TSCourse/chapter_06_probabilistic_prediction/lessons/L03_bsts_crps_calibration.md)

[Walkthrough](../TSCourse/chapter_06_probabilistic_prediction/walkthroughs/lesson_three_intervals.ipynb) |
[Challenge - Regime-Change Coverage](../TSCourse/chapter_06_probabilistic_prediction/challenges/challenge1_regime_coverage.ipynb)

### (7) Deep Learning for Sequences & Trajectories

```
Tag: done

Author: Cfir Hadar
```

* [Compressed Sequence-Model Refresher](../TSCourse/chapter_07_deep_learning_sequences/lessons/L01_sequence_models.md)
* [Transformer Forecasters & the DLinear Lens](../TSCourse/chapter_07_deep_learning_sequences/lessons/L02_transformers_and_dlinear.md)
* [Trajectory Prediction Proper](../TSCourse/chapter_07_deep_learning_sequences/lessons/L03_trajectory_prediction.md)

[Walkthrough](../TSCourse/chapter_07_deep_learning_sequences/walkthroughs/lesson_dlinear_vs_patch.ipynb)

### (8) Frontier: Foundation Models & Modern SSMs

```
Tag: done

Author: Cfir Hadar
```

* [Foundation Models for Time Series](../TSCourse/chapter_08_frontier_foundation_models/lessons/L01_foundation_models.md)
* [State-Space Models as Deep Architectures](../TSCourse/chapter_08_frontier_foundation_models/lessons/L02_deep_ssms.md)

### (9) Capstone: Critical Reading & Reproduction

```
Tag: done

Author: Cfir Hadar
```

* [Reading TS/Trajectory Papers Critically](../TSCourse/chapter_09_capstone/lessons/L01_critical_reading.md)

[Capstone - Reproduce a Central Claim](../TSCourse/chapter_09_capstone/challenges/capstone_reproduction.ipynb)
