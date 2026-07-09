```
Author: Cfir Hadar

Tags: Done
```
# Lesson 03 - Deep Learning for Time Series

## Motivation

Classical models (Lessons 01-02) excel when you have one series (or a few), short history, and roughly linear dynamics. Deep models earn their complexity in the complementary regime: **many related series** (thousands of sensors, products, tracks), long histories, nonlinear dynamics, and rich covariates (spatial coordinates, calendar, categorical metadata). The key shift in mindset is from *local* to **global** modeling: instead of one model per series, train **one network across all series**, letting it share statistical strength — a cold-start series can be forecast well because the model has seen thousands like it.

Before any architecture: deep models do *not* remove the discipline of Lesson 01. Split by time, backtest with rolling origins, normalize per-series (e.g. scale each window by its mean — networks are not scale-invariant), and always compare against a seasonal-naive and an ARIMA/ETS baseline. A surprising fraction of published "deep forecasting wins" evaporate against tuned classical baselines.

## Problem Types (not everything is forecasting)

The same backbones serve several distinct tasks — know which one you are solving:

* **Forecasting**: predict $y_{t+1:t+H}$ from the past (point or probabilistic).
* **Classification / regression of whole series**: e.g. classify a vehicle type from its track, or activity from accelerometer data.
* **Anomaly detection**: flag sensor faults / unusual behavior; typically via forecasting error or reconstruction error (an autoencoder over windows) — directly relevant to detecting **sensor errors** in a fleet of sensors.
* **Imputation / denoising**: fill gaps and clean glitches — the deep sibling of Kalman smoothing.

## Architectures

**RNN/LSTM (seen in DS101 Lesson 04).** The natural sequential model: a learned nonlinear state update $h_t=\phi(h_{t-1},x_t)$ — literally a *learned, nonlinear Kalman-style state*, but without calibrated uncertainty unless you add it. DeepAR (Amazon) is the canonical industrial pattern: a global LSTM trained on many series that outputs *distribution parameters* (e.g. $\mu_t,\sigma_t$) and is trained by maximum likelihood → probabilistic forecasts by ancestral sampling. Weaknesses: sequential (slow) training, limited effective memory.

**Temporal Convolutional Networks (TCN).** 1D CNNs adapted to time: **causal** convolutions (pad only the past — never let the network peek ahead of the forecast origin) with **dilations** doubling per layer ($1,2,4,8,\dots$), so the receptive field grows exponentially with depth,

$$
RF=1+\sum_{l}(k-1)\,d_l \approx 1 + (k-1)(2^{L}-1).
$$

Fully parallel training, stable gradients, strong defaults for both forecasting and series classification; WaveNet is the famous instance.

**Transformers (Chapter 5, Lesson 05).** Causal self-attention over time steps: direct access to any past event regardless of distance (e.g. "the same hour last week") at $O(n^2)$ cost. Positional encodings naturally carry timestamps and calendar features. A cautionary result to know: on standard long-horizon benchmarks, embarrassingly simple linear models (DLinear) beat many elaborate time-series transformers — recent designs (PatchTST: attend over *patches* of time steps; iTransformer: attend *across variables*) fixed much of this. The honest current picture: transformers win with lots of data and long contexts; N-BEATS/N-HiTS (pure MLP stacks) and TCNs remain extremely competitive; and pretrained **time-series foundation models** (TimesFM, Chronos, Moirai) now offer decent zero-shot forecasts — worth trying before training anything.

## Spatial Structure: Tracks, Geography, Sensor Networks

The user-facing problems this course cares about — moving objects, geographic tracks, arrays of sensors — add a *spatial* dimension to the temporal one. The main patterns:

* **Trajectories as sequences.** A track is a sequence of $(x_t, y_t, \dot x_t, \dot y_t, \dots)$ or of local displacements. Feed to RNN/TCN/transformer for trajectory prediction or classification. Practical details that matter more than architecture: represent positions in a *local* frame (deltas relative to current position/heading, not raw lat/lon — removes meaningless absolute-coordinate variance); handle irregular sampling explicitly (include $\Delta t$ as a feature, or use interpolation); for lat/lon distances use the haversine metric, or project to a local planar frame first.
* **Geographic covariates.** Static spatial features (terrain, road type, zone embeddings — e.g. geohash embedding vectors) join the per-step inputs of a global model; this is how one network serves many locations.
* **Sensor arrays / networks: Spatio-Temporal GNNs.** When $N$ sensors interact (traffic sensors on a road network, weather stations, a distributed array), model them as a graph: nodes = sensors, edges = proximity/correlation. Each layer alternates **graph message passing across sensors** with **temporal modeling along time** (GNN + TCN/GRU: DCRNN, Graph WaveNet, STGCN). The graph enforces that information flows along physical structure — the spatial analog of a CNN's locality prior. A sensor's failure also becomes detectable as *inconsistency with its graph neighbors*, a strictly stronger signal than its own history alone.
* **Sensor errors, revisited.** The deep version of Lesson 02's toolkit: train a model of normal multi-sensor behavior (forecasting or masked reconstruction), flag sensors whose residuals are large *relative to what neighbors imply*, and impute their values from the graph. Hybrids are often best in practice — a Kalman filter with learned components (learned dynamics/noise models, e.g. KalmanNet) keeps the calibrated recursion and adds nonlinearity.

## Choosing, Honestly

| Regime | Reach for |
| --- | --- |
| One/few series, short history | ETS/ARIMA (Lesson 01), gradient boosting on lag features |
| Known dynamics + noisy sensors, real-time | Kalman family (Lesson 02) |
| Many related series, covariates | Global deep model: DeepAR-style LSTM, TCN, N-HiTS, PatchTST |
| Series classification (tracks, activities) | TCN / 1D-CNN first; transformer if data is plentiful |
| Networked sensors, spatial dependence | Spatio-temporal GNN |
| No time to train | Pretrained TS foundation model, then the table above |

And in every row: a seasonal-naive baseline, per-series normalization, and time-respecting backtests are non-negotiable.

## Walkthrough

Revisit [LSTM Stock-Price Prediction](../../../Datascience101/chapter_03_machine_learning_2/walkthroughs/lesson4_lstm_stock-price-prediction.ipynb) with this lesson's eyes: identify the windowing, normalization, and evaluation choices, and ask which of the pitfalls above it avoids. Then do the [Sensor Fusion challenge](../challenges/challenge1_sensor_fusion.ipynb), which combines Lessons 02 and 03.
