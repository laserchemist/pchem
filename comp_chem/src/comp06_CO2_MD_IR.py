# LAB
# %% [markdown]
# # Computational Lab 6: Molecular Dynamics on the Learned Surface, and the IR Spectrum of CO₂
# *CHEM 3376 — Physical Chemistry Laboratory: Data Analysis & Computation*
#
# Now the nuclei move. **Classical molecular dynamics** solves Newton's equations
#
# $$m_i\,\ddot{\mathbf x}_i = \mathbf F_i = -\nabla_i E(\mathbf x)$$
#
# step by step, using the machine-learned surface from Lab 5 for $E$. Nothing in MD is harmonic: the molecule stretches, bends,
# rotates, and trades energy between modes exactly as the surface dictates. You will:
#
# 1. compute **forces** from the surface;
# 2. write the **velocity Verlet** integrator and test it by **energy conservation**;
# 3. choose a **time step**;
# 4. run a trajectory and **watch it in 3-D**;
# 5. compute the **IR spectrum** from the dipole moment along the trajectory, and compare it with the harmonic sticks of Lab 4;
# 6. average over many trajectories to see the **rotational P and R branches**, and heat the molecule to 1500 K.
#
# ### Why the dipole gives the spectrum
# A molecule absorbs light at frequency ω when its dipole oscillates at ω. The absorption spectrum is the Fourier transform of the
# **dipole autocorrelation function** $\langle\boldsymbol\mu(0)\cdot\boldsymbol\mu(t)\rangle$. In practice (the Wiener–Khinchin theorem)
# this is $|\hat{\boldsymbol\mu}(\omega)|^2$, the squared Fourier transform of $\boldsymbol\mu(t)$ itself
# ([McQuarrie, *Statistical Mechanics*, Ch. 21](https://uscibooks.aip.org/books/statistical-mechanics/)).
#
# This lab uses the surface and dipole model you saved in Lab 5 (or a reference copy). The longer runs take 1–3 minutes each.
#
# <br>**<center>Learning Goals**
#
# |Area|Concept|
# |---|---|
# |Classical mechanics|Forces as the gradient of the potential; Newton's equations; conservation of energy and momentum|
# |Numerical methods|The velocity Verlet integrator; time-step error (∝ Δt²) and stability; finite-difference gradients|
# |Statistical mechanics|Thermal initial conditions (equipartition, Maxwell–Boltzmann); averaging over trajectories|
# |Spectroscopy|The IR spectrum as the Fourier transform of μ(t); selection rules beyond the harmonic approximation; P and R branches; temperature effects|
# |Python|Fast array operations; `np.fft.rfft`; window functions; animations|

# %% [markdown]
# ---
# ## 📄 Abstract
#
# <div style="border:2px solid #1f4e79; border-radius:8px; padding:10px 14px; background:#f3f7fb; color:#1a1a1a;">
#
# **Write this section last**, after you have finished the lab. Your abstract goes in the cell below, replacing `ANSWER`.
#
# In **150–250 words**, as in a journal article: state the purpose; describe the method (classical MD with velocity Verlet on the
# Gaussian-process surface, the time step and its test, thermal sampling, the spectrum from the dipole); give your **key numbers**
# (energy conservation, the MD band positions against the harmonic values, the P–R separation, the changes at 1500 K); and finish
# with what MD shows that the harmonic picture cannot.
# </div>

# %% [markdown] answer
# Classical molecular dynamics of CO₂ was run on a Gaussian-process potential energy surface fitted to B3LYP/def2-SVP energies, to compute the infrared spectrum without the harmonic approximation. Forces were obtained by central finite differences of the learned energy, and Newton's equations were integrated with velocity Verlet. The total energy fluctuates by 0.9 meV with a 1 fs step and grows as Δt² (slope 1.9), so a 0.5 fs step (one twenty-seventh of the 13.5 fs ν₃ period) was used. A 10 ps trajectory at 300 K, started from a thermal distribution of mode amplitudes and Maxwell–Boltzmann velocities, shows the bend and asymmetric stretch together with overall rotation. The spectrum, the Fourier transform of the dipole moment from a bond-charge model, has bands at 651 and 2481 cm⁻¹, close to the harmonic 654 and 2467 cm⁻¹, while the symmetric stretch is absent, as symmetry requires at any amplitude. Averaging 24 trajectories gives the ν₃ band its rotational P and R branches, separated by about 20 cm⁻¹ (classical estimate 26 cm⁻¹). At 1500 K the bands broaden and shift to lower wavenumber because the larger-amplitude motion samples the softer, anharmonic parts of the surface. MD captures anharmonicity, mode coupling and rotation, which the harmonic picture leaves out, but as a classical method it has no zero-point energy.

# %% [markdown]
# ---

# %% [markdown]
# ## ⚙️ Initialization — DO NOT EDIT
# The next two cells load the course tools (auto-checks, quizzes, styling) and record the lab version. **Do not change them** — with one exception:
#
# 👁️ **Accessible view (optional):** set `ACCESSIBLE_VIEW = True` in the cell below for larger, more readable text and colorblind-safe plots. You can switch back any time by setting it to `False` and re-running the cell (then *Kernel → Restart* to clear the text styling).
#
# 👉 **Run the cell below first** (Shift+Enter) before anything else in this lab.

# %%
# 🔧 Run this cell first! (Shift+Enter)
ACCESSIBLE_VIEW = True     # True = larger text and colorblind-safe plots

import sys, os
for _up in ("..", os.path.join("..", ".."), os.path.join("..", "..", "..")):
    _shared = os.path.abspath(os.path.join(os.getcwd(), _up, "shared-public"))
    if os.path.isdir(_shared):
        sys.path.insert(0, _shared)
        break
if ACCESSIBLE_VIEW:
    from init_access import *
else:
    from init import *

# %%
# 🔖 Lab version — do not edit. Used by the automated checks below.
ver = '2026Comp06V1'

# %% [markdown]
# ### 🧰 Standard imports
# `numpy`, `matplotlib` and `labkit` (constants, the 3-D viewer and animations). No quantum chemistry in this lab: every energy
# comes from the learned surface. The last lines load this lab's **answer checks**.

# %%
# Standard imports for this lab
import time, pickle
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
for _up in ("..", os.path.join("..", "..")):          # labkit is in comp_chem/, the folder above this lab
    if os.path.isdir(os.path.join(_up, "labkit")):
        sys.path.insert(0, os.path.abspath(_up))
        break
import labkit as lk

# Answer checks for this lab: the checker (pchem_check.py) and the answer key (answers_comp06.json) are in this folder
from pchem_check import Checker
lab = Checker('comp06', 'answers_comp06.json')

# %% [markdown]
# Now run the cell below to enter your name.

# %%
# Enter your name as a string, e.g. name = 'Linus Pauling'
name = 'Instructor'  # SOLUTION

# %% [markdown]
# **Just run the cell below.** It loads your surface from Lab 5 (or refits the reference surface from the provided data, about
# 2 s) and defines `energy(X)` (eV) and `dipole(X)` (D) for one geometry `(3, 3)` or many `(n, 3, 3)`.

# %%
# Just run this cell: the learned energy surface and dipole model
def features(X):
    """(r1, r2, cos θ) for CO2 geometries X (..., 3, 3), atom order O, C, O (as in Lab 5)."""
    b1 = X[..., 0, :] - X[..., 1, :]
    b2 = X[..., 2, :] - X[..., 1, :]
    r1 = np.linalg.norm(b1, axis=-1); r2 = np.linalg.norm(b2, axis=-1)
    return np.stack([r1, r2, np.sum(b1 * b2, axis=-1) / (r1 * r2)], axis=-1)

_mine = os.path.join(os.path.dirname(lk.data_path("x")), "..", "results", "co2_ml_model.pkl")
if os.path.exists(_mine):
    model = pickle.load(open(_mine, "rb"))
    print("loaded your surface from Lab 5")
else:   # fallback: refit the reference surface from the provided data
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    d = lk.load_co2_samples(); F = features(d["X"])
    Fs = np.concatenate([F, F[:, [1, 0, 2]]]); Es = np.concatenate([d["dE_eV"], d["dE_eV"]])
    mn, sc = Fs.mean(0), Fs.std(0)
    gp = GaussianProcessRegressor(ConstantKernel(1.0, (1e-3, 1e7)) * RBF([1.0] * 3, (1e-2, 1e5)) + WhiteKernel(1e-5, (1e-6, 1e-2)),
                                  normalize_y=True).fit((Fs - mn) / sc, Es)
    h = np.load(lk.data_path("co2_harmonic.npz"))
    model = {"gp": gp, "f_mean": mn, "f_scale": sc, "c_dip": np.array([-0.2714, -0.1549, 0.6024, 0.0340]),
             "r_e": h["X0"][2, 2], "X0": h["X0"], "symbols": ["O", "C", "O"], "masses": h["masses"], "wn_dft": h["wn"]}
    print("Lab 5 surface not found in results/: refitted the reference surface")

gp, f_mean, f_scale, c_dip, r_e = model["gp"], model["f_mean"], model["f_scale"], model["c_dip"], model["r_e"]
X0, symbols, wn_harm = model["X0"], model["symbols"], model["wn_dft"]
masses = np.asarray(model["masses"])[:, None]          # shape (3, 1), so it multiplies x, y and z of each atom

def energy(X):
    """ML energy (eV) for one geometry (3, 3) or many (n, 3, 3)."""
    F = features(np.asarray(X)).reshape(-1, 3)
    return gp.predict((F - f_mean) / f_scale).reshape(np.shape(X)[:-2])

def dipole(X):
    """Bond-charge dipole model (debye) for geometries (..., 3, 3)."""
    X = np.asarray(X); F = features(X)
    phi = lambda F: np.stack([np.ones(F.shape[:-1]), F[..., 0] - r_e, F[..., 1] - r_e, 1 + F[..., 2]], -1)
    q1, q2 = phi(F) @ c_dip, phi(F[..., [1, 0, 2]]) @ c_dip
    return lk.DEBYE_PER_E_ANG * (q1[..., None] * (X[..., 0, :] - X[..., 1, :]) + q2[..., None] * (X[..., 2, :] - X[..., 1, :]))

print("E(X0) =", round(float(energy(X0)), 6), "eV;  dipole(X0) =", np.round(dipole(X0), 6), "D")

# %% [markdown]
# ## 1. Forces

# %% [markdown]
# ### <font color=blue> **Question 1.1.** </font>
# The force is minus the gradient of the energy. The simplest way to get it is a **central finite difference** in each of the 9
# Cartesian coordinates:
#
# $$F_k = -\frac{E(\mathbf x + h\,\mathbf e_k) - E(\mathbf x - h\,\mathbf e_k)}{2h}.$$
#
# For speed, the function builds all 18 displaced geometries as one array `P` of shape `(18, 3, 3)`: `P[2k]` is coordinate k moved
# by +h and `P[2k+1]` by −h. One call to `energy` gives all 18 energies `e`. Complete the `return` line: `e[0::2]` are the +h energies
# and `e[1::2]` the −h energies.
#
# The cell then checks two laws of physics: at the minimum the forces are ≈ 0, and at any geometry the **total** force on the molecule
# is zero (Newton's third law: the energy doesn't change when the whole molecule moves).
#
# 🐍 **Python tip — slicing with a step.** `e[0::2]` takes every second element starting at 0 (elements 0, 2, 4, ...); `e[1::2]`
# starts at 1.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `return -((e[0::2] - e[1::2]) / (2 * h)).reshape(3, 3)`.
# </details>

# %%
def forces(X, h=1e-3):
    """Forces (eV/Å), shape (3, 3), on the atoms of geometry X."""
    P = np.repeat(X[None], 18, axis=0)
    for k in range(9):
        P[2 * k].flat[k] += h
        P[2 * k + 1].flat[k] -= h
    e = energy(P)
    return -((e[0::2] - e[1::2]) / (2 * h)).reshape(3, 3)  # SOLUTION

Xt = X0 + np.random.default_rng(0).normal(scale=0.03, size=(3, 3))
print("max |F| at the minimum:", np.abs(forces(X0)).max(), "eV/Å")
print("total force at a distorted geometry:", forces(Xt).sum(0), "eV/Å")

# %%
lab.check('1.1')

# %% [markdown]
# ## 2. The Velocity Verlet Integrator

# %% [markdown]
# ### <font color=blue> **Question 2.1.** </font>
# With acceleration $\mathbf a = \mathbf F/m$ (times `lk.ACC` for units of Å/fs² when F is in eV/Å and m in amu), one step of length
# $\Delta t$ is
#
# 1. $\mathbf v \leftarrow \mathbf v + \tfrac12 \Delta t\,\mathbf a(\mathbf x)$ — half a kick
# 2. $\mathbf x \leftarrow \mathbf x + \Delta t\,\mathbf v$ — a drift
# 3. compute $\mathbf a(\mathbf x)$ at the new position
# 4. $\mathbf v \leftarrow \mathbf v + \tfrac12 \Delta t\,\mathbf a(\mathbf x)$ — the other half kick
#
# It is time-reversible and conserves energy very well over long runs, which is why almost every MD code uses it
# ([Frenkel & Smit, Ch. 4](https://doi.org/10.1016/B978-0-12-267351-1.X5000-7)). Complete the three update lines in `run_md`.
# The check runs 200 fs and confirms that the total energy stays constant.
#
# 🐍 **Python tip — `+=`.** `V += a` means `V = V + a`: it updates the array in place.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `V += 0.5 * dt * F / masses * lk.ACC` (twice) and `X += dt * V`.
# </details>

# %%
def kinetic(V):
    """Kinetic energy (eV) from velocities V (Å/fs)."""
    return 0.5 * np.sum(masses * V**2) / lk.ACC

def run_md(X, V, dt=0.5, n_steps=1000, save_every=1):
    """Velocity Verlet. Returns traj (n_saved, 3, 3), dipoles (n_saved, 3), total energies (n_saved,)."""
    X, V = X.copy(), V.copy()
    traj, mus, etot = [], [], []
    F = forces(X)
    for step in range(n_steps):
        V += 0.5 * dt * F / masses * lk.ACC  # SOLUTION
        X += dt * V  # SOLUTION
        F = forces(X)
        V += 0.5 * dt * F / masses * lk.ACC  # SOLUTION
        if step % save_every == 0:
            traj.append(X.copy()); mus.append(dipole(X)); etot.append(energy(X) + kinetic(V))
    return np.array(traj), np.array(mus), np.array(etot)

# %% [markdown]
# **Just run the cell below.** For a thermal start, each normal mode gets a random harmonic displacement with
# $\langle\tfrac12\omega^2Q^2\rangle = \tfrac12 k_BT$, and each atom random **Maxwell–Boltzmann** velocities,
# $v \sim \mathcal N(0, \sqrt{k_BT/m})$ per direction. The centre-of-mass velocity is removed so the molecule doesn't drift away.

# %%
# Just run this cell: thermal starting conditions, and a first 200 fs test of your integrator
h_data = np.load(lk.find_result("co2_harmonic.npz"))
modes, omega = h_data["modes"], 2 * np.pi * lk.C_CM_PER_FS * h_data["wn"]
m3 = np.repeat(masses.ravel(), 3)

def thermal_start(T, rng):
    kT = lk.KB_EV * T
    Q = rng.normal(size=4) * np.sqrt(kT * lk.ACC) / omega
    X = X0 + ((modes @ Q) / np.sqrt(m3)).reshape(3, 3)
    V = rng.normal(size=(3, 3)) * np.sqrt(kT * lk.ACC / masses)
    V -= (masses * V).sum(0) / masses.sum()
    return X, V

X_test, V_test = thermal_start(300, np.random.default_rng(1))
_, _, et_test = run_md(X_test, V_test, dt=0.5, n_steps=400)
print(f"200 fs at 300 K: total energy changes by at most {1000 * np.ptp(et_test):.3f} meV "
      f"(kinetic energy at the start {1000 * kinetic(V_test):.1f} meV)")

# %%
lab.check('2.1')

# %% [markdown]
# ## 3. Choosing the Time Step

# %% [markdown]
# ### <font color=blue> **Question 3.1.** </font>
# The fastest vibration (ν₃ ≈ 2470 cm⁻¹) has a period of 13.5 fs. **Just run the first cell below** (about 30 s): it runs 1 ps from
# the same 300 K start with Δt = 4, 2, 1, 0.5 and 0.25 fs, and plots $E_\text{tot}(t) - E_\text{tot}(0)$ and the size of the energy
# fluctuation (`np.ptp`, max − min) against Δt on log–log axes.
#
# Then, in the second cell, find the **slope** of log(fluctuation) against log(Δt) for Δt = 0.5, 1 and 2 fs (`dts[1:4]` and
# `spread[1:4]`), with `np.polyfit(..., 1)`, which returns (slope, intercept).
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `slope = np.polyfit(np.log(dts[1:4]), np.log(spread[1:4]), 1)[0]`.
# </details>

# %%
# Just run this cell: energy conservation for five time steps
rng = np.random.default_rng(7)
Xs, Vs = thermal_start(300, rng)
t0 = time.time()
dts, spread = np.array([4.0, 2.0, 1.0, 0.5, 0.25]), []
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for dt in dts:                      # largest first, so the small-step curves are drawn on top
    every = max(1, int(1 / dt))
    _, _, et = run_md(Xs, Vs, dt, int(1000 / dt), save_every=every)
    ax[0].plot(np.arange(len(et)) * every * dt, (et - et[0]) * 1000, label=f"Δt = {dt} fs")
    spread.append(np.ptp(et) * 1000)
spread = np.array(spread)
ax[0].set_ylim(-6, 6); ax[0].set_xlabel("t / fs"); ax[0].set_ylabel("E_tot − E_tot(0) / meV"); ax[0].legend(fontsize=8)
ax[1].loglog(dts, spread, "o-"); ax[1].loglog(dts, spread[3] * (dts / dts[3])**2, "k--", label="∝ Δt²")
ax[1].set_xticks(dts, [str(d) for d in dts]); ax[1].minorticks_off()
ax[1].set_xlabel("Δt / fs"); ax[1].set_ylabel("energy fluctuation / meV"); ax[1].legend()
plt.tight_layout(); plt.show()
print(f"({time.time() - t0:.0f} s)  fluctuation (meV):", dict(zip(dts, np.round(spread, 3))))

# %%
slope = np.polyfit(np.log(dts[1:4]), np.log(spread[1:4]), 1)[0]  # SOLUTION
print(f"slope = {slope:.2f}")

# %%
lab.check('3.1')

# %% [markdown]
# ### <font color=blue> **Question 3.2 Discussion.** </font>
# <font color=green>
#
# 1. Which time steps are acceptable, and why does the energy blow up when Δt is a sizeable fraction of the vibrational period?
# 2. What does your slope say about how the error of velocity Verlet depends on Δt?
# 3. What is the cost of a smaller time step? Which step would you choose for a molecule containing hydrogen (C–H stretch near 3000 cm⁻¹)?
# </font>
#
# ---

# %% [markdown] answer
# Δt ≤ 1 fs is acceptable (fluctuation 0.9 meV at 1 fs and 0.27 meV at 0.5 fs); at 2 fs it is 3.6 meV, and at 4 fs (≈ period/3) 43 meV, effectively unstable. Verlet assumes the force is nearly constant over a step, which fails when the step is a large fraction of the fastest period; rule of thumb Δt ≲ period/10–20. The slope near 2 shows that the error grows as Δt² (Verlet is a second-order method). The cost grows as 1/Δt: halving the step doubles the run time. A C–H stretch has a period of about 11 fs, and hydrogen moves fast, so 0.25 fs is the safe choice.

# %% [markdown]
# ---

# %%
lab.check('3.2d')

# %% [markdown]
# ## 4. A Trajectory, and the Movie
# **Just run the two cells below.** The first runs 10 ps at 300 K with Δt = 0.5 fs (under a minute) and plots $r_1(t)$, $r_2(t)$ and the
# O–C–O angle for the first 500 fs. The second plays the first 2 ps in 3-D: drag to rotate, scroll to zoom.

# %%
# Just run this cell: 10 ps at 300 K
rng = np.random.default_rng(11)
X_init, V_init = thermal_start(300, rng)
t0 = time.time()
traj, mu, et = run_md(X_init, V_init, dt=0.5, n_steps=20000, save_every=1)
print(f"10 ps in {time.time() - t0:.0f} s; energy range {1000 * np.ptp(et):.2f} meV")
times = np.arange(len(traj)) * 0.5
Fq = features(traj)
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(times[:1000], Fq[:1000, 0], label="r₁"); ax[0].plot(times[:1000], Fq[:1000, 1], label="r₂")
ax[0].set_ylabel("C–O / Å"); ax[0].legend()
ax[1].plot(times[:1000], np.degrees(np.arccos(np.clip(Fq[:1000, 2], -1, 1)))); ax[1].set_ylabel("O–C–O / deg")
ax[1].set_xlabel("t / fs"); plt.show()

# %%
# Just run this cell: the movie, every 2nd step (1 fs per frame) for the first 2 ps
lk.animate_trajectory(symbols, traj[:4000], stride=2, interval=30)

# %% [markdown]
# ### <font color=blue> **Question 4.1 Discussion.** </font>
# <font color=green>
#
# In the plots, which motions can you identify, and with roughly what periods? (Compare with the harmonic wavenumbers: period =
# 1/(c ν̃) = 33 356 fs / ν̃[cm⁻¹].) Do r₁ and r₂ move together or opposite? In the movie, what does the molecule do besides vibrate?
# </font>
#
# ---

# %% [markdown] answer
# r₁ and r₂ oscillate fastest, mostly in opposite phase, with a period of about 13.5 fs (the asymmetric stretch, 2470 cm⁻¹), plus a slower in-phase modulation (the symmetric stretch, 24 fs). The angle oscillates with a period of about 51 fs (the bend, 654 cm⁻¹) around an average below 180°, because a thermally excited bend is never exactly linear. In the movie the molecule also tumbles end over end: overall rotation, with a period of a few picoseconds at 300 K.

# %% [markdown]
# ---

# %%
lab.check('4.1d')

# %% [markdown]
# ## 5. The Dipole and the IR Spectrum
# **Just run the cell below.** It plots the three dipole components for the first 500 fs.

# %%
# Just run this cell: the dipole moment along the trajectory
for k, comp in enumerate("xyz"):
    plt.plot(times[:1000], mu[:1000, k], label=f"μ_{comp}")
plt.xlabel("t / fs"); plt.ylabel("μ / D"); plt.legend(); plt.show()

# %% [markdown]
# ### <font color=blue> **Question 5.1.** </font>
# Complete `ir_spectrum(mu, dt_fs)`. The first steps are given:
# 1. subtract the mean dipole;
# 2. multiply by a **Hann window** (`np.hanning`), so the sharp ends of the record don't add ripples;
# 3. Fourier transform each component with `np.fft.rfft` (padded to 4× the length for a smoother curve) and sum $|\hat\mu|^2$ over x, y, z.
#
# You write:
# 4. the frequency axis `nu` in cm⁻¹: `np.fft.rfftfreq(n, d=dt_fs)` gives frequencies in fs⁻¹; divide by c (`lk.C_CM_PER_FS`);
# 5. the return line: the spectrum `S` multiplied by $\tilde\nu^2$ (the classical absorption line shape).
#
# The check tests your function on a pure 1000 cm⁻¹ oscillation, so you know what the answer must be.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `nu = np.fft.rfftfreq(n, d=dt_fs) / lk.C_CM_PER_FS` and `return nu, S * nu**2`.
# </details>

# %%
def ir_spectrum(mu, dt_fs):
    """Classical IR absorption spectrum from a dipole time series. Returns (wavenumber in cm⁻¹, intensity, arbitrary units)."""
    m = (mu - mu.mean(0)) * np.hanning(len(mu))[:, None]
    n = 4 * len(m)
    S = np.sum(np.abs(np.fft.rfft(m, n=n, axis=0))**2, axis=1)
    nu = np.fft.rfftfreq(n, d=dt_fs) / lk.C_CM_PER_FS  # SOLUTION
    return nu, S * nu**2  # SOLUTION

nu, I = ir_spectrum(mu, 0.5)
sel = nu < 3000
plt.figure(figsize=(8, 3.5))
plt.plot(nu[sel], I[sel] / I[sel].max(), label="MD, 300 K, one 10 ps trajectory")
plt.vlines(wn_harm, 0, 1, color="r", ls=":", label="harmonic (Lab 4)")
plt.xlabel("wavenumber / cm⁻¹"); plt.ylabel("absorbance (normalised)"); plt.legend(); plt.show()

# %%
lab.check('5.1')

# %% [markdown]
# ### <font color=blue> **Question 5.2.** </font>
# Find the peak of each band in your spectrum: in each window `w`, the wavenumber where `I` is largest.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# Inside the loop: `peaks[band] = nu[w][np.argmax(I[w])]`.
# </details>

# %%
windows = {"bend": (500, 800), "symmetric stretch": (1200, 1500), "asymmetric stretch": (2200, 2700)}
peaks, rel = {}, {}
for band, (lo, hi) in windows.items():
    w = (nu > lo) & (nu < hi)
    peaks[band] = nu[w][np.argmax(I[w])]  # SOLUTION
    rel[band] = I[w].max() / I[sel].max()
    print(f"{band:19s}: peak at {peaks[band]:6.0f} cm⁻¹, relative intensity {rel[band]:.1e}")

# %%
lab.check('5.2')

# %% [markdown]
# ### <font color=blue> **Question 5.3 Discussion.** </font>
# <font color=green>
#
# Which normal modes appear in the MD spectrum and which does not? Is this the same conclusion as the harmonic analysis of Lab 4,
# even though nothing in MD is harmonic? Why do the MD peaks sit slightly away from the harmonic sticks?
# </font>
#
# ---

# %% [markdown] answer
# The bend (651 cm⁻¹) and the asymmetric stretch (2481 cm⁻¹) appear; the symmetric stretch (≈1389 cm⁻¹) is absent (relative intensity ~10⁻⁸ or less, numerical noise). The selection rule is a symmetry statement that holds at any amplitude, not a harmonic one: the symmetric stretch never breaks the inversion centre. The peaks shift a little from the sticks because the surface is anharmonic and modes couple (the bend is thermally excited at 300 K), because classical MD has no zero-point energy (so it gives near-harmonic band positions at low T, not quantum fundamentals), and because a 10 ps record limits the resolution to about 3 cm⁻¹. A single trajectory also has one fixed rotation, which can split a band into two peaks.

# %% [markdown]
# ---

# %%
lab.check('5.3d')

# %% [markdown]
# ## 6. Rotation Shapes the Band: Averaging over Trajectories
# A single molecule rotates at one rate, which splits each band into two peaks (rotation adds to and subtracts from the vibrational
# frequency). A real gas has a thermal distribution of rotational speeds. **Just run the cell below** (under 2 minutes): it averages
# the spectra of 24 independent 300 K trajectories (4 ps each) and zooms in on the asymmetric stretch. You should see the **P and R
# branches** of the gas-phase spectrum ([Atkins, Topic 11D](https://global.oup.com/academic/product/atkins-physical-chemistry-9780198769866)).

# %%
# Just run this cell: 24 trajectories at 300 K, averaged
def averaged_spectrum(T, n_traj=24, t_ps=4, dt=0.5, seed=0):
    rng = np.random.default_rng(seed)
    acc = 0
    for k in range(n_traj):
        X, V = thermal_start(T, rng)
        _, m, _ = run_md(X, V, dt, int(t_ps * 1000 / dt), save_every=2)
        nu_k, I_k = ir_spectrum(m, 2 * dt)
        acc = acc + I_k
    return nu_k, acc / n_traj

t0 = time.time()
nu300, I300 = averaged_spectrum(300)
print(f"({time.time() - t0:.0f} s)")
w = (nu300 > 2400) & (nu300 < 2550)
plt.plot(nu300[w], I300[w] / I300[w].max()); plt.axvline(wn_harm[3], color="r", ls=":", label="harmonic ν₃")
plt.xlabel("wavenumber / cm⁻¹"); plt.title("ν₃ band, 300 K, 24 trajectories"); plt.legend(); plt.show()

# %% [markdown]
# ### <font color=blue> **Question 6.1.** </font>
# For a linear molecule the maxima of the classical P and R branches are separated by about $\Delta\tilde\nu_{PR} = 2.36\sqrt{B\,T}$ cm⁻¹,
# with the rotational constant $B$ in cm⁻¹ and $T$ in K. For CO₂, $B$ = 0.39 cm⁻¹. Compute `dPR_expected` at 300 K and compare it with
# the separation of the two maxima in your plot.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `dPR_expected = 2.36 * np.sqrt(0.39 * 300)`.
# </details>

# %%
dPR_expected = 2.36 * np.sqrt(0.39 * 300)  # SOLUTION
print(f"expected P–R separation at 300 K: {dPR_expected:.0f} cm⁻¹")

# %%
lab.check('6.1')

# %% [markdown]
# ## 7. A Hot Molecule
# **Just run the two cells below** (under 2 minutes). They repeat the average at 1500 K, compare the bend and asymmetric-stretch
# bands at the two temperatures, and make the movie for the hot molecule.

# %%
# Just run this cell: 24 trajectories at 1500 K, compared with 300 K
t0 = time.time()
nu1500, I1500 = averaged_spectrum(1500)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for nu_, I_, lbl in ((nu300, I300, "300 K"), (nu1500, I1500, "1500 K")):
    for a, (lo, hi) in zip(ax, ((550, 760), (2300, 2560))):
        w = (nu_ > lo) & (nu_ < hi)
        a.plot(nu_[w], I_[w] / I_[w].max(), label=lbl)
for a, k in zip(ax, (0, 3)):
    a.axvline(wn_harm[k], color="r", ls=":"); a.set_xlabel("wavenumber / cm⁻¹"); a.legend()
ax[0].set_title("bend ν₂"); ax[1].set_title("asymmetric stretch ν₃"); plt.show()
print(f"({time.time() - t0:.0f} s)")

# %%
# Just run this cell: the hot molecule
X, V = thermal_start(1500, np.random.default_rng(3))
hot, _, _ = run_md(X, V, 0.5, 4000, save_every=1)
lk.animate_trajectory(symbols, hot, stride=2, interval=30)

# %% [markdown]
# ### <font color=blue> **Question 7.1 Discussion.** </font>
# <font color=green>
#
# How do the bands change at 1500 K: position, width, and shape of the P and R branches? Explain each change in terms of the surface
# (anharmonicity, mode coupling) and of rotation. Is the learned surface still trustworthy at 1500 K? (Recall the sampling energy of Lab 5.)
# </font>
#
# ---

# %% [markdown] answer
# At 1500 K both bands broaden and shift to lower wavenumber (ν₃ spreads over roughly 2320–2450 cm⁻¹): with larger amplitudes the molecule samples the softer, anharmonic parts of the surface, and the excited bend couples to the stretches. The P and R branches move apart and broaden, because the rotational distribution extends to higher J (Δν̃_PR ∝ √T). At 1500 K each mode holds about kT = 0.13 eV on average, but the tails of the distribution approach the 0.6 eV sampled in Lab 5, so the surface is near the edge of its training data; I would check by computing DFT energies for some hot-trajectory frames.

# %% [markdown]
# ---

# %%
lab.check('7.1d')

# %% [markdown]
# ## 8. Save a Figure and the Trajectory
# **Just run the cell below.** A static film strip is useful for your report. The `.xyz` file can be opened in other viewers
# (VMD, Avogadro, Jmol).

# %%
# Just run this cell: film strip and .xyz file
fig, ax = plt.subplots(figsize=(9, 1.8))
lk.trajectory_strip(symbols, traj, times, frames=[0, 10, 20, 30, 40, 50], plane=(2, 0), ax=ax)
plt.show()
with open(lk.results_path("co2_300K.xyz"), "w") as f:
    for i in range(0, 4000, 2):
        f.write(f"3\nt = {times[i]:.1f} fs\n")
        for s, x in zip(symbols, traj[i]):
            f.write(f"{s} {x[0]:.5f} {x[1]:.5f} {x[2]:.5f}\n")
print("wrote results/co2_300K.xyz")

# %% [markdown]
# ### 📋 Summary of your results
# **Just run the cell below.**

# %%
# Just run this cell: summary table
summary = Table().with_columns(
    'Quantity', ['energy fluctuation, Δt = 0.5 fs (meV)', 'energy fluctuation, Δt = 1 fs (meV)', 'slope of log(error) vs log(Δt)',
                 'MD bend peak (cm⁻¹)', 'MD asymmetric-stretch peak (cm⁻¹)', 'symmetric stretch, relative intensity',
                 'expected P–R separation, 300 K (cm⁻¹)'],
    'Value', [round(spread[3], 3), round(spread[2], 3), round(slope, 2), round(peaks["bend"]), round(peaks["asymmetric stretch"]),
              f"{rel['symmetric stretch']:.0e}", round(dPR_expected, 1)],
    'Harmonic (Lab 4) / experiment', ['', '', '2 (Verlet)', f"{wn_harm[0]:.0f} / 667", f"{wn_harm[3]:.0f} / 2349", '0 (symmetry)', ''])
summary.show()

# %% [markdown]
# ### 📝 Quick Check — Molecular Dynamics
# Test what you've learned. This is just for practice — answer as many times as you like, it isn't recorded.

# %%
with open("questionsComp06.json", "r") as f:
    display_quiz(json.load(f), shuffle_answers=True)

# %% [markdown]
# ---
# ## 9. Conclusions

# %% [markdown]
# ### <font color=blue> **Question 9. Conclusions.** </font>
# <div style="border:2px solid #1f4e79; border-radius:8px; padding:10px 14px; background:#f3f7fb; color:#1a1a1a;">
#
# Write the **Conclusions** of this lab in the cell below (one or two paragraphs, about 150–300 words), using **your own numbers**
# from the summary table. Address:
#
# 1. How you know the dynamics are numerically sound.
# 2. What the MD spectrum shares with the harmonic spectrum of Lab 4, and what it adds (band shapes, shifts, temperature).
# 3. What classical MD cannot capture, and how that limits the comparison with experiment.
# 4. How this whole workflow (Labs 4–6) explains why CO₂ is a greenhouse gas.
# </div>
#
# ---

# %% [markdown] answer
# The dynamics are sound: the forces vanish at the minimum and sum to zero at any geometry, and with Δt = 0.5 fs the total energy fluctuates by well under 1 meV, with the error growing as Δt² (slope ≈ 2), as expected for velocity Verlet. The MD spectrum keeps the harmonic selection rule exactly, bend and asymmetric stretch bright and symmetric stretch dark, because that rule comes from symmetry, not from the harmonic approximation; and its bands (651 and 2481 cm⁻¹) lie within a few cm⁻¹ of the harmonic values. What MD adds is the shape of the bands: the P and R branches from thermal rotation (about 20 cm⁻¹ apart against the 26 cm⁻¹ classical estimate), and the broadening and red shift at 1500 K from anharmonicity and mode coupling. Classical MD has no zero-point energy and no quantised rotational levels, so it reproduces band envelopes rather than individual lines, and its band positions correspond to harmonic frequencies rather than to the observed fundamentals; its accuracy is also capped by B3LYP/def2-SVP (ν₃ about 5 % high). Together, Labs 4–6 show that CO₂ absorbs infrared light because its bend and asymmetric stretch break its inversion symmetry, and that the bend absorbs in a thermally broadened band centred near 667 cm⁻¹, right under the Earth's thermal emission.

# %% [markdown]
# ---

# %% [markdown]
# ## 10. Reflection

# %% [markdown]
# ### <font color=blue> **Question 10.** </font>
# At the end of each lab, please include a reflection.
# * How did this lab go? Was it clear what each calculation was doing?
# * Which part of the Python was new to you, and which part was most useful?
# * Did you use the hints? Which questions needed them?
# * Were there questions you found especially challenging that you would like your instructor to review in class?
# * How long did the lab take you to complete?
#
# **Use the markdown cell below this one and write your reflection on this lab.**
#
# ---

# %% [markdown] answer
# (Instructor version: students write their reflection here.) Seeing the spectrum emerge from a trajectory tied the course together.

# %% [markdown]
# ---

# %% [markdown]
# #### Save notebook before check
# This cell checks your **Abstract** (at the top of the notebook), your **Conclusions** and your **Reflection**.

# %%
lab.check('abstract')
lab.check('9c')
lab.check('10d')

# %% [markdown]
# ## 📚 References
# 1. D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed., Academic Press (2002), Ch. 4 (molecular dynamics, the Verlet algorithm), [doi:10.1016/B978-0-12-267351-1.X5000-7](https://doi.org/10.1016/B978-0-12-267351-1.X5000-7).
# 2. W. C. Swope, H. C. Andersen, P. H. Berens and K. R. Wilson, "A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules," *J. Chem. Phys.* **76**, 637 (1982), [doi:10.1063/1.442716](https://doi.org/10.1063/1.442716) (velocity Verlet).
# 3. D. A. McQuarrie, *Statistical Mechanics*, University Science Books (2000), Ch. 21 (time-correlation functions and spectra).
# 4. M. Thomas, M. Brehm, R. Fligg, P. Vöhringer and B. Kirchner, "Computing vibrational spectra from ab initio molecular dynamics," *Phys. Chem. Chem. Phys.* **15**, 6608 (2013), [doi:10.1039/c3cp44302g](https://doi.org/10.1039/c3cp44302g).
# 5. P. Atkins, J. de Paula and J. Keeler, *Atkins' Physical Chemistry*, 11th ed., Oxford University Press (2018): Topic 11B (rotational spectra), Topic 11D (vibration–rotation spectra of polyatomic molecules, P and R branches).
# 6. G. Herzberg, *Molecular Spectra and Molecular Structure II. Infrared and Raman Spectra of Polyatomic Molecules*, Van Nostrand (1945) (band contours of linear molecules).
# 7. NIST Chemistry WebBook, "Carbon dioxide": [IR spectrum](https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Type=IR-SPEC&Index=1).

# %% [markdown]
# ## 🏁 Final Step — Check, Submit, Then Upload to Canvas
#
# Follow these steps **in order**:
#
# 1. 💾 **Save** the notebook (Ctrl/Cmd + S): the written-answer checks and the submission read the *saved* file.
# 2. ✅ **Run the cell below** to see your score: one row per question, with the number of attempts. Any question still marked ❌ or
#    *not checked*: go back, fix it, and run its check again.
# 3. 📨 **Click Submit Lab.** This records your score, your abstract, your conclusions and your discussion answers under your
#    JupyterHub username. You can submit again after making changes; the latest submission counts.
# 4. 📤 **Make a PDF and upload it to Canvas.** Export an HTML copy ([instructions](https://laserchemist.github.io/data/submission-instructions/export-html)), open it in your browser, and use **Print → Save as PDF**. Check that your plots, tables and written answers all appear in the PDF, then upload it to this lab's Canvas assignment.

# %%
# Your score for this lab, and the Submit button
lab.score()
lab.submit(name, lab_id='Comp06')
