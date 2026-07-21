"""Regenerate the TSCourse lesson figures that have no suitable equivalent on Wikimedia Commons.

Run from anywhere: python make_figures.py  (writes into this directory).
Figures sourced from Commons instead are the wm_* files - see CREDITS.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = str(__import__("pathlib").Path(__file__).parent) + "/"
plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
R = np.random.default_rng(0)


def save(name):
    plt.tight_layout()
    plt.savefig(OUT + name, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close()


def track(n=200, seed=1):
    """A synthetic 2D flight track: cruise, turn, cruise."""
    rng = np.random.default_rng(seed)
    hdg = np.concatenate([np.zeros(60), np.linspace(0, 1.1, 40), np.full(50, 1.1),
                          np.linspace(1.1, -.4, 30), np.full(n - 180, -.4)])
    v = 1.0
    pos = np.cumsum(np.c_[v * np.cos(hdg), v * np.sin(hdg)], 0)
    return pos + rng.normal(0, .3, pos.shape), hdg


# 1 - five problem framings on one track ---------------------------------
pos, hdg = track()
fig, ax = plt.subplots(1, 5, figsize=(11, 2.4), sharex=True, sharey=True)
titles = ["Forecasting\n(future positions)", "Filtering\n(state now)", "Smoothing\n(best past estimate)",
          "Classification\n(cruise vs. turn)", "Anomaly detection\n(is this track odd?)"]
for a, t in zip(ax, titles):
    a.plot(*pos.T, lw=.8, color="0.7")
    a.set_title(t, fontsize=7.5)
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
ax[0].plot(*pos[150:].T, lw=2, color="tab:red")
ax[1].plot(*pos[149], "o", color="tab:blue", ms=6)
ax[1].add_patch(plt.Circle(pos[149], 8, fill=False, color="tab:blue"))
ax[2].plot(*pos[:150].T, lw=2, color="tab:green")
ax[3].plot(*pos[60:100].T, lw=2, color="tab:orange")
ax[3].plot(*pos[150:180].T, lw=2, color="tab:orange")
ax[3].plot(*pos[:60].T, lw=2, color="tab:blue")
ax[3].plot(*pos[100:150].T, lw=2, color="tab:blue")
bad = pos.copy(); bad[120:] += np.c_[np.linspace(0, 25, 80), np.linspace(0, -25, 80)]
ax[4].plot(*bad.T, lw=1.4, color="tab:red", ls="--")
save("problem_framings.png")

# 2 - stationarity -------------------------------------------------------
n = 300
e = R.normal(0, 1, n)
rw = np.cumsum(e)
ll = np.zeros(n); mu = 0.
for t in range(n):          # local level: mean-reverting-ish state + noise
    mu = .98 * mu + R.normal(0, .5); ll[t] = mu + R.normal(0, .3)
fig, ax = plt.subplots(1, 3, figsize=(10, 2.4))
ax[0].plot(rw, lw=.9); ax[0].set_title("Random walk: no fixed mean, variance grows $\\propto t$")
ax[1].plot(ll, lw=.9, color="tab:green"); ax[1].set_title("Local level: wanders, but bounded")
ax[2].plot(np.diff(rw), lw=.7, color="tab:red"); ax[2].set_title("Differenced random walk: white noise")
save("stationarity.png")

# 4 - temporal evaluation protocols --------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(7.5, 3.4), sharex=True)
N = 40
for k in range(4):                                   # rolling origin
    tr, te = 10 + 5 * k, 5
    ax[0].barh(k, tr, color="tab:blue"); ax[0].barh(k, te, left=tr, color="tab:orange")
ax[0].set_title("Walk-forward / rolling origin: train always precedes test", loc="left")
for k in range(4):                                   # blocked with purge
    s = 9 * k
    ax[1].barh(k, 6, left=s, color="tab:orange"); ax[1].barh(k, 1.5, left=s + 6, color="0.6")
    ax[1].barh(k, N - s - 7.5, left=s + 7.5, color="tab:blue"); ax[1].barh(k, s, color="tab:blue")
ax[1].set_title("Blocked CV with an embargo (grey) between folds", loc="left")
idx = R.permutation(N)                               # random k-fold (wrong)
cols = np.where(idx < 10, "tab:orange", "tab:blue")
ax[2].bar(np.arange(N), 1, color=cols, width=1)
ax[2].set_title("Random k-fold: test points sit *before* training points — leakage", loc="left")
for a in ax:
    a.set_yticks([]); a.grid(False)
ax[2].set_xlabel("time →")
save("evaluation_protocols.png")

# 6 - what each filter can represent -------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(10, 2.3))
xx, yy = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
ax[0].contourf(xx, yy, np.exp(-(xx**2 + yy**2 / 2) / 2), 8, cmap="Blues")
ax[0].set_title("KF: exact Gaussian\n(linear-Gaussian)", fontsize=7.5)
ax[1].contourf(xx, yy, np.exp(-(xx**2 + yy**2 / 2) / 2), 8, cmap="Blues")
th = np.linspace(0, 2 * np.pi, 100)
ax[1].plot(1.6 * np.cos(th) - .6 * np.sin(th) ** 2, 1.9 * np.sin(th), "k--", lw=.8)
ax[1].set_title("EKF: Gaussian fitted to a\nlinearization (dashed = truth)", fontsize=7.5)
sp = np.array([[0, 0], [1.8, 0], [-1.8, 0], [0, 1.3], [0, -1.3]])
ax[2].contourf(xx, yy, np.exp(-(xx**2 + yy**2 / 2) / 2), 8, cmap="Blues")
ax[2].plot(*sp.T, "ko", ms=4); ax[2].set_title("UKF: sigma points pushed\nthrough the true map", fontsize=7.5)
p1 = R.normal([-1.4, .8], .45, (300, 2)); p2 = R.normal([1.5, -.6], .35, (200, 2))
ax[3].scatter(*np.vstack([p1, p2]).T, s=2, alpha=.5, color="tab:purple")
ax[3].set_title("PF: weighted samples —\nmultimodal, non-Gaussian", fontsize=7.5)
for a in ax:
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(-3, 3); a.set_ylim(-3, 3)
save("filter_ladder.png")

# 7 - motion models & IMM ------------------------------------------------
dt, T = 1., 90
def sim(kind):
    p, v = np.zeros((T, 2)), np.array([1., 0.])
    for t in range(1, T):
        if kind == "CA": v = v * 1.02
        if kind == "CT" and 30 <= t < 60:
            w = .06; c, s = np.cos(w), np.sin(w); v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        p[t] = p[t - 1] + v * dt
    return p
fig, ax = plt.subplots(1, 2, figsize=(8, 2.6))
for k, c in zip(["CV", "CA", "CT"], ["tab:blue", "tab:green", "tab:red"]):
    ax[0].plot(*sim(k).T, color=c, label=k)
ax[0].legend(fontsize=7); ax[0].set_title("Constant velocity / acceleration / coordinated turn")
mu_ct = 1 / (1 + np.exp(-(np.minimum(np.arange(T) - 30, 60 - np.arange(T))) / 3))
ax[1].plot(mu_ct, color="tab:red", label="P(turn model)")
ax[1].plot(1 - mu_ct, color="tab:blue", label="P(cruise model)")
ax[1].axvspan(30, 60, color="0.85"); ax[1].legend(fontsize=7)
ax[1].set_title("IMM mode probabilities track the maneuver"); ax[1].set_xlabel("t")
save("motion_models.png")

# 8 - data association ---------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
pred = np.array([0., 0.]); truth = np.array([1.1, .5])
clutter = R.uniform(-3, 3, (9, 2))
for a, t in zip(ax, ["Nearest neighbour:\nclutter steals the track",
                     "Gated NN: $\\chi^2$ gate rejects\nimplausible measurements",
                     "JPDA: weighted average of\nall gated measurements"]):
    a.plot(*clutter.T, "kx", ms=5, label="clutter")
    a.plot(*truth, "go", label="true detection")
    a.plot(*pred, "b^", ms=8, label="prediction")
    a.set_title(t, fontsize=7.5); a.set_xticks([]); a.set_yticks([])
d = clutter[np.argmin(np.linalg.norm(clutter - pred, axis=1))]
ax[0].annotate("", d, pred, arrowprops=dict(arrowstyle="->", color="r"))
for a in ax[1:]:
    a.add_patch(plt.Circle(pred, 1.6, fill=False, ls="--", color="b"))
ax[1].annotate("", truth, pred, arrowprops=dict(arrowstyle="->", color="g"))
ing = clutter[np.linalg.norm(clutter - pred, axis=1) < 1.6]
for c in ing:
    ax[2].annotate("", c, pred, arrowprops=dict(arrowstyle="->", color="0.6", alpha=.7))
ax[2].annotate("", truth, pred, arrowprops=dict(arrowstyle="->", color="g"))
ax[0].legend(fontsize=6.5, loc="lower left")
save("data_association.png")

# 9 - resampling artifacts ----------------------------------------------
pos, _ = track(seed=3)
keep = np.sort(R.choice(200, 45, replace=False))
fig, ax = plt.subplots(1, 3, figsize=(10, 2.6), sharex=True, sharey=True)
ax[0].plot(*pos.T, lw=.7, color="0.75"); ax[0].plot(*pos[keep].T, "o", ms=2.5, color="tab:blue")
ax[0].set_title("Irregular samples over the true path", fontsize=8)
lin = np.array([np.interp(np.arange(200), keep, pos[keep, i]) for i in range(2)]).T
ax[1].plot(*pos.T, lw=.7, color="0.75"); ax[1].plot(*lin.T, lw=1.2, color="tab:red")
ax[1].set_title("Linear interpolation: fabricated\ncorners, understated speed", fontsize=8)
k = np.ones(11) / 11
sm = np.array([np.convolve(lin[:, i], k, "same") for i in range(2)]).T[6:-6]
ax[2].plot(*pos.T, lw=.7, color="0.75"); ax[2].plot(*sm.T, lw=1.2, color="tab:green")
ax[2].set_title("Over-smoothing: the turn\nis flattened away", fontsize=8)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
save("resampling_artifacts.png")

# 10 - segmentation ------------------------------------------------------
seg = np.concatenate([R.normal(0, .3, 120), R.normal(2.5, .3, 90), R.normal(.7, .8, 140)])
cps = [120, 210]
fig, ax = plt.subplots(figsize=(7.5, 2.2))
ax.plot(seg, lw=.7, color="0.4")
for c in cps:
    ax.axvline(c, color="tab:red", ls="--")
for a, b, lbl in [(0, 120, "cruise"), (120, 210, "climb"), (210, 350, "maneuvering")]:
    ax.hlines(seg[a:b].mean(), a, b, color="tab:blue", lw=2)
    ax.text((a + b) / 2, 3.4, lbl, ha="center", fontsize=7)
ax.set_title("Change-point segmentation: piecewise-constant model, penalised number of segments", fontsize=8)
save("segmentation.png")

# 12 - intervals & coverage under regime change --------------------------
n = 240
sig = np.where(np.arange(n) < 150, .5, 1.8)
y = np.sin(np.arange(n) / 12) + R.normal(0, 1, n) * sig
fig, ax = plt.subplots(1, 2, figsize=(9, 2.6))
mu = np.sin(np.arange(n) / 12)
ax[0].plot(y, ".", ms=2, color="0.5")
ax[0].fill_between(np.arange(n), mu - 1.96 * .5, mu + 1.96 * .5, alpha=.3, color="tab:blue",
                   label="split conformal (fixed width)")
ad = np.where(np.arange(n) < 150, .5, np.clip((np.arange(n) - 150) / 25, 0, 1) * 1.3 + .5)
ax[0].plot(mu + 1.96 * ad, color="tab:red", lw=1, label="adaptive / online conformal")
ax[0].plot(mu - 1.96 * ad, color="tab:red", lw=1)
ax[0].axvline(150, color="k", ls=":"); ax[0].legend(fontsize=6.5)
ax[0].set_title("Regime change at t=150: fixed width silently under-covers", fontsize=8)
w = 20
cov_f = [np.mean(np.abs(y[i:i + w] - mu[i:i + w]) < 1.96 * .5) for i in range(0, n - w, w)]
cov_a = [np.mean(np.abs(y[i:i + w] - mu[i:i + w]) < 1.96 * ad[i:i + w]) for i in range(0, n - w, w)]
xs = np.arange(0, n - w, w)
ax[1].plot(xs, cov_f, "o-", color="tab:blue", label="fixed"); ax[1].plot(xs, cov_a, "s-", color="tab:red", label="adaptive")
ax[1].axhline(.95, ls="--", color="k"); ax[1].axvline(150, color="k", ls=":")
ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=7)
ax[1].set_title("Rolling empirical coverage vs. the 95% target", fontsize=8)
save("coverage_regime.png")

# 13 - patching ----------------------------------------------------------
s = np.sin(np.arange(96) / 7) + .4 * R.normal(0, 1, 96)
fig, ax = plt.subplots(2, 1, figsize=(7.5, 2.8), sharex=True)
ax[0].plot(s, lw=.8, color="0.4"); ax[0].plot(s, "o", ms=2.5, color="tab:blue")
ax[0].set_title("Point-wise tokens: 96 tokens, each nearly information-free", fontsize=8)
for k in range(6):
    ax[1].axvspan(k * 16, (k + 1) * 16 - .5, color=f"C{k}", alpha=.18)
ax[1].plot(s, lw=.9, color="0.4")
ax[1].set_title("Patches: 6 tokens with local shape — cheaper attention, better inductive bias", fontsize=8)
save("patching.png")

# 14 - multimodal futures ------------------------------------------------
hist = np.c_[np.arange(30), np.zeros(30)]
fig, ax = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
for a in ax:
    a.plot(*hist.T, "k-", lw=1.6, label="history")
f = np.arange(30, 60)
mean = np.c_[f, np.zeros(30)]
ax[0].plot(*mean.T, color="tab:red", lw=1.6, label="mean prediction")
ax[0].fill_between(f, -(f - 30) * .12, (f - 30) * .12, alpha=.25, color="tab:red")
ax[0].set_title("Unimodal regression: averages the modes\ninto a physically impossible path", fontsize=8)
for sgn, c in [(1, "tab:blue"), (0, "tab:green"), (-1, "tab:orange")]:
    ax[1].plot(f, sgn * ((f - 30) ** 1.7) * .03, color=c, lw=1.5)
    for _ in range(12):
        ax[1].plot(f, sgn * ((f - 30) ** 1.7) * .03 + R.normal(0, .05) * (f - 30), color=c, alpha=.15, lw=.6)
ax[1].set_title("Mixture / CVAE: separate modes\n(turn left, straight, turn right)", fontsize=8)
ax[0].legend(fontsize=7)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
save("multimodal_futures.png")

print("figures written")
