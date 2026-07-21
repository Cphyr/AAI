```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Coordinate Frames & Kinematic Features

## Motivation

A trajectory is not a pair of numeric columns. Latitude and longitude are angles on a curved
surface, in units that are not comparable to each other, and treating them as $x,y$ produces
distances that are wrong by up to a factor of two at high latitude, headings that are wrong
everywhere, and filters whose $Q$ means nothing. Getting the frame right takes an hour; getting it
wrong contaminates every downstream number.

## Frames

| Frame | Coordinates | Use for |
| --- | --- | --- |
| **Geodetic** (WGS-84) | $(\varphi,\lambda,h)$ — lat, lon, height | storage, exchange, plotting on maps |
| **ECEF** | $(x,y,z)$ metres, Earth-centred | global geometry, satellites, converting between frames |
| **Local tangent plane / ENU** | east, north, up in metres about a local origin | *all* filtering and feature work over a limited area |
| **Body frame** | forward, right, down w.r.t. the platform | IMU/sensor data, attitude |

The working rule: **filter and featurise in a local ENU frame, store in geodetic.** Pick an origin
per track (or per region), convert once, and your motion models, $Q$ in m²/s³, distances and
velocities all become ordinary Euclidean quantities. Over hundreds of kilometres, use a projection
with acceptable distortion for your area (UTM, or an azimuthal equidistant projection centred on
the origin) and know its error budget.

Distances on the sphere, when you need them: the **haversine** formula for great-circle distance,
Vincenty/Karney for ellipsoidal accuracy. Never Euclidean distance on raw degrees, and never
compare a degree of longitude to a degree of latitude ($\cos\varphi$ says they differ).

## Kinematic features

From a clean, time-sorted segment $\{(t_k,p_k)\}$ in a metric frame:

$$
v_k=\frac{p_{k+1}-p_{k}}{\Delta t_k},\quad
s_k=\|v_k\|,\quad
\chi_k=\operatorname{atan2}(v_{k,E},\,v_{k,N}),\quad
a_k=\frac{v_{k+1}-v_k}{\Delta t_k}
$$

* **Heading** $\chi$ is circular: unwrap it before differencing, and never average headings
  arithmetically — use $\operatorname{atan2}(\overline{\sin\chi},\overline{\cos\chi})$, or work with
  $(\sin\chi,\cos\chi)$ as two features.
* **Turn rate** $\omega_k=\mathrm{wrap}(\chi_{k+1}-\chi_k)/\Delta t_k$, wrapped to $(-\pi,\pi]$ —
  the single most informative trajectory feature there is.
* **Curvature** $\kappa=\dfrac{|\dot p_x\ddot p_y-\dot p_y\ddot p_x|}{(\dot p_x^2+\dot p_y^2)^{3/2}}$,
  which unlike turn rate is speed-independent (geometry, not kinematics). Numerically fragile at
  low speed — guard the denominator.
* **Acceleration**: split into tangential ($\hat v\cdot a$, speeding up) and normal
  ($\|a - (\hat v\cdot a)\hat v\|$, turning). They mean different things physically and behave
  differently as features.
* **Vertical**: climb rate, and its own sign changes — cheap and highly discriminative for aircraft.
* **Aggregate / shape**: path straightness $\|p_T-p_1\|/\sum_k\|\Delta p_k\|$, radius of gyration,
  bounding-box aspect, total heading change, number of stops, dwell time inside regions,
  speed and turn-rate quantiles (quantiles beat means — they survive outliers and describe the
  distribution).

Numerical differentiation amplifies noise: at 1 Hz with 25 m position error, the naive velocity
error is $\approx35$ m/s. Differentiate the **filtered** state (the Kalman filter estimates
velocity directly and correctly), or use a Savitzky-Golay derivative, never raw finite differences
on noisy positions. This is the single most common bug in trajectory feature code.

## Invariances — choose them deliberately

Ask what your task should *not* depend on, then build it in:

* Translation (absolute position) → use relative/displacement features, or a per-track origin.
* Rotation (absolute heading) → use speed, turn rate, curvature, not raw heading.
* Time origin and duration → resample to fixed length, or use duration-normalised features.
* Speed scaling → normalise, if two platforms differ only in speed and that is irrelevant.

But be careful: discarding absolute position also discards geography, and for many real tasks
("does this track approach the restricted area?") geography is the entire signal. Invariance is a
claim about the problem, not a free improvement.

## Assumptions & failure modes

| Assumption | Breaks when | Symptom | Response |
| --- | --- | --- | --- |
| Lat/lon behave like metres | always; worse near the poles | distances and headings wrong; filters mistuned | convert to ENU first |
| Finite differences give velocity | noisy positions | velocity noise dominates; turn rate is pure noise | differentiate filtered states |
| Heading is a normal number | wrap-around at $\pm\pi$ | spurious 360°/s turn rates; nonsense means | wrap and unwrap explicitly; use $(\sin,\cos)$ |
| Features are frame-independent | mixed tracks over a wide area | model learns the region, not the behaviour | per-track local frame; test on a held-out region |
| Curvature is stable | near-zero speed | division by ~0, huge spikes | guard with a speed threshold |

**Lens check:** lens 1 (this lesson *is* representation) with lens 3 in the numerical-fragility
rows.

## Next

[Lesson 03 - Segmentation of Long Recordings](L03_segmentation.md)
