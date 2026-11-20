# LAB
# %% [markdown]
# # Computational Lab 5: A Machine-Learned Potential Energy Surface for CO₂
# *CHEM 3376 — Physical Chemistry Laboratory: Data Analysis & Computation*
#
# The harmonic picture of Lab 4 is exact only for infinitesimal vibrations. To see how a real, warm molecule moves
# (anharmonicity, rotation, energy flowing between modes) you will run **molecular dynamics** in Lab 6. MD needs the energy
# and forces at every time step: 20 000 steps × 0.7 s per DFT calculation is about 4 hours. Instead you will **learn** the
# surface from a few hundred DFT calculations, and evaluate it in microseconds.
#
# The method is **Gaussian process regression (GPR)**: it predicts the energy at a new geometry as a smooth, weighted
# combination of the energies at the training geometries, weighted by how *similar* the geometries are. It also gives an
# **uncertainty**, which tells you when it is guessing. Four principles carry over from research-grade machine-learned surfaces:
#
# 1. **Build the physics into the inputs.** The energy cannot change if the molecule is translated or rotated, or if the two
#    identical O atoms are swapped. Use coordinates that have those symmetries.
# 2. **Sample where the dynamics will go**, a little beyond, and not much further.
# 3. **Judge the fit on data it has never seen** (a held-out test set), not on its training error.
# 4. **Check physical quantities**, such as the harmonic frequencies, against the reference calculation.
#
# This lab uses the normal modes you saved in Lab 4 (or a reference copy).
#
# <br>**<center>Learning Goals**
#
# |Area|Concept|
# |---|---|
# |Potential energy surfaces|Sampling along normal modes; internal coordinates; permutation symmetry|
# |Machine learning|Gaussian process regression; kernels and length scales; training vs held-out test sets; learning curves; extrapolation|
# |Spectroscopy|Harmonic frequencies as a physical test of a surface; a bond-charge model of the dipole moment|
# |Statistics|RMS error; predictive uncertainty; linear least squares|
# |Python|Random numbers with `np.random.default_rng`; `np.concatenate`; `np.linalg.lstsq`; scikit-learn|

# %% [markdown]
# ---
# ## 📄 Abstract
#
# <div style="border:2px solid #1f4e79; border-radius:8px; padding:10px 14px; background:#f3f7fb; color:#1a1a1a;">
#
# **Write this section last**, after you have finished the lab. Your abstract goes in the cell below, replacing `ANSWER`.
#
# In **150–250 words**, as in a journal article: state the purpose; describe the method (normal-mode sampling, B3LYP/def2-SVP
# reference data, the features, Gaussian process regression, the bond-charge dipole model); give your **key numbers** (test RMS
# errors for the energy and dipole, the learning-curve trend, the GP harmonic frequencies against DFT); say where the model fails;
# and finish with whether the surface is good enough for molecular dynamics.
# </div>

# %% [markdown] answer
# A machine-learned potential energy surface and dipole model for CO₂ were built so that molecular dynamics can be run without a DFT calculation at every step. Geometries were sampled by distributing up to 0.6 eV of harmonic energy randomly over the four normal modes, and B3LYP/def2-SVP energies and dipoles were computed for 100 test geometries; 320 geometries sampled the same way formed the training set. The energy was learned by Gaussian process regression on the symmetry-adapted features (r₁, r₂, cos θ), with every training point also entered with r₁ and r₂ swapped to impose the O ↔ O permutation symmetry. On the held-out test set the RMS energy error is 0.02 meV (0.2 cm⁻¹), and it falls from 0.2 meV with 20 training points to a floor set by the numerical noise of the DFT grid. The GP's own uncertainty (mean σ = 0.17 meV) is cautious, about eight times the actual error. Harmonic frequencies of the learned surface agree with the DFT Hessian to within 5 cm⁻¹. Along a single-bond stretch the GP follows DFT inside the training range but fails outside it, where its predicted uncertainty grows. A four-parameter bond-charge model with geometry-dependent charges (q(O) = −0.27 e at equilibrium) reproduces the test dipoles to about 0.01 D, and its charge flux explains the intensity of the asymmetric stretch. The surface and dipole are accurate enough for molecular dynamics at the energies sampled.

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
ver = '2026Comp05V1'

# %% [markdown]
# ### 🧰 Standard imports
# `numpy`, `matplotlib`, PySCF, scikit-learn's Gaussian process, and `labkit`. The last lines load this lab's **answer checks**.

# %%
# Standard imports for this lab
import time, pickle
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from pyscf import gto, dft
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
for _up in ("..", os.path.join("..", "..")):          # labkit is in comp_chem/, the folder above this lab
    if os.path.isdir(os.path.join(_up, "labkit")):
        sys.path.insert(0, os.path.abspath(_up))
        break
import labkit as lk

# Answer checks for this lab: the checker (pchem_check.py) and the answer key (answers_comp05.json) are in this folder
from pchem_check import Checker
lab = Checker('comp05', 'answers_comp05.json')

# %% [markdown]
# Now run the cell below to enter your name.

# %%
# Enter your name as a string, e.g. name = 'Linus Pauling'
name = 'Instructor'  # SOLUTION

# %% [markdown]
# **Just run the cell below.** It loads the geometry and normal modes from Lab 4 and defines `co2_energy_dipole(X)`, which
# returns the B3LYP energy (eV) and dipole (D) at a geometry `X`.

# %%
# Just run this cell: Lab 4 results and the DFT reference calculation
h = np.load(lk.find_result("co2_harmonic.npz"))
symbols = [str(s) for s in h["symbols"]]
X0, wn_dft, modes, masses = h["X0"], h["wn"], h["modes"], h["masses"]
m3 = np.repeat(masses, 3)
print("harmonic DFT wavenumbers (cm⁻¹):", np.round(wn_dft, 1))

def co2_energy_dipole(X, basis="def2-svp", xc="b3lyp"):
    """B3LYP energy (eV) and dipole vector (D) of CO2 at geometry X (3 x 3, Å)."""
    mol = gto.M(atom=[[s, tuple(x)] for s, x in zip(symbols, X)], basis=basis, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc
    e = mf.kernel() * lk.HARTREE_TO_EV
    return e, mf.dip_moment(unit="Debye", verbose=0)

E_min, _ = co2_energy_dipole(X0)
print(f"E(X0) = {E_min:.4f} eV")

# %% [markdown]
# ## 1. Sampling Geometries

# %% [markdown]
# ### 1.1 Displacing along the normal modes
# A good way to generate realistic distorted geometries is to displace along the **normal modes**. If mode $k$ (mass-weighted
# coordinate $Q_k$, angular frequency $\omega_k$) holds energy $\varepsilon_k$, then $\varepsilon_k = \tfrac12\omega_k^2 Q_k^2$, so its
# harmonic amplitude is
#
# $$ Q_k = \frac{\sqrt{2\varepsilon_k}}{\omega_k} , $$
#
# and the Cartesian displacement is $\Delta\mathbf x = \sum_k Q_k\,\mathbf L_k/\sqrt{m}$.
#
# **Units.** With $\varepsilon$ in eV, $\omega$ in rad/fs and $Q$ in Å·amu$^{1/2}$, $\tfrac12\omega^2 Q^2$ comes out in amu·Å²/fs², which is
# $1/\texttt{lk.ACC}$ eV. So $Q_k = \sqrt{2\varepsilon_k \times \texttt{lk.ACC}}\,/\,\omega_k$.

# %% [markdown]
# ### <font color=blue> **Question 1.1.** </font>
# Complete `sample_geometry(rng, E_max)`. It should
# - draw a total energy `E_tot` between 0 and `E_max` (eV): `E_max * rng.random()`;
# - split it randomly among the 4 modes: `eps = E_tot * rng.dirichlet(np.ones(4))` (given);
# - compute the four amplitudes `Q` with a random sign each (`rng.choice([-1, 1], 4)` is given as `signs`).
#
# The check projects your geometry back onto the normal modes and confirms that it holds exactly `E_tot` of harmonic energy.
#
# 🐍 **Python tip — random numbers.** `rng = np.random.default_rng(2024)` makes a random-number generator with a fixed **seed**,
# so the "random" numbers are the same every time you run the notebook. `rng.random()` is a number between 0 and 1.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `E_tot = E_max * rng.random()` and `Q = np.sqrt(2 * eps * lk.ACC) / omega * signs`.
# </details>

# %%
omega = 2 * np.pi * lk.C_CM_PER_FS * wn_dft          # angular frequencies, rad/fs

def sample_geometry(rng, E_max=0.6):
    """Random geometry with up to E_max (eV) of harmonic energy spread over the four normal modes."""
    E_tot = E_max * rng.random()  # SOLUTION
    eps = E_tot * rng.dirichlet(np.ones(4))          # energy in each mode (eV), adding up to E_tot
    signs = rng.choice([-1, 1], 4)
    Q = np.sqrt(2 * eps * lk.ACC) / omega * signs  # SOLUTION
    dx = (modes @ Q) / np.sqrt(m3)                   # Cartesian displacement, 9 numbers
    return X0 + dx.reshape(3, 3)

rng = np.random.default_rng(2024)
print(np.round(sample_geometry(rng), 3))

# %%
lab.check('1.1')

# %% [markdown]
# ### <font color=blue> **Question 1.2.** </font>
# Compute B3LYP energies and dipoles for **100** of your own geometries (about 1 minute). These are your **test set**. For
# training you will use 320 geometries computed the same way, which are provided so that everyone has the same training set.
# Complete the line for `E_test`, the energies **relative to the minimum** `E_min` (eV).
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `ED` is a list of (energy, dipole) pairs: `E_test = np.array([e for e, _ in ED]) - E_min`.
# </details>

# %%
N_TEST = 100                     # use 30 if you are short of time
t = time.time()
X_test = np.array([sample_geometry(rng) for _ in range(N_TEST)])
ED = [co2_energy_dipole(X) for X in X_test]
E_test = np.array([e for e, _ in ED]) - E_min  # SOLUTION
mu_test = np.array([m for _, m in ED])
print(f"{N_TEST} DFT points in {time.time() - t:.0f} s;  energies {E_test.min():.3f} to {E_test.max():.3f} eV")

train = lk.load_co2_samples()
X_train, E_train, mu_train = train["X"], train["dE_eV"], train["dipole_D"]
print("training set:", X_train.shape[0], "points,", train["method"])

# %%
lab.check('1.2')

# %% [markdown]
# ## 2. Symmetry-Adapted Features

# %% [markdown]
# ### 2.1 Why not Cartesian coordinates?
# A model fed the 9 Cartesian coordinates would have to *learn* that moving or rotating the molecule changes nothing. Instead,
# give it three internal coordinates that already have those symmetries: the two C–O distances $r_1, r_2$ and the cosine of the
# O–C–O angle. With bond vectors $\mathbf b_1 = \mathbf x_{O_1} - \mathbf x_C$ and $\mathbf b_2 = \mathbf x_{O_2} - \mathbf x_C$,
#
# $$ \cos\theta = \frac{\mathbf b_1\cdot\mathbf b_2}{r_1\, r_2} , $$
#
# which is −1 for the linear molecule. ($\cos\theta$ is better than $\theta$ itself: θ has a kink at 180°, where the molecule
# bends through linear.) The O ↔ O swap is handled in Section 3 by entering every training point twice.

# %% [markdown]
# ### <font color=blue> **Question 2.1.** </font>
# Complete `features(X)` with the line for `cos`. It works on one geometry (shape `(3, 3)`) or many (shape `(n, 3, 3)`), so use
# `np.sum(..., axis=-1)` for the dot product over x, y, z.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `cos = np.sum(b1 * b2, axis=-1) / (r1 * r2)`.
# </details>

# %%
def features(X):
    """(r1, r2, cos θ) for CO2 geometries X of shape (..., 3, 3), atom order O, C, O."""
    b1 = X[..., 0, :] - X[..., 1, :]                 # C → O1
    b2 = X[..., 2, :] - X[..., 1, :]                 # C → O2
    r1 = np.linalg.norm(b1, axis=-1)
    r2 = np.linalg.norm(b2, axis=-1)
    cos = np.sum(b1 * b2, axis=-1) / (r1 * r2)  # SOLUTION
    return np.stack([r1, r2, cos], axis=-1)

print("features at X0:", np.round(features(X0), 4))
F_train, F_test = features(X_train), features(X_test)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(F_train[:, 0], F_train[:, 1], ".", label="training"); ax[0].plot(F_test[:, 0], F_test[:, 1], "x", label="your test set")
ax[0].set_xlabel("r₁ / Å"); ax[0].set_ylabel("r₂ / Å"); ax[0].legend()
ax[1].hist(np.degrees(np.arccos(np.clip(F_train[:, 2], -1, 1))), 30); ax[1].set_xlabel("O–C–O angle / deg")
plt.tight_layout(); plt.show()

# %%
lab.check('2.1')

# %% [markdown]
# ## 3. Fitting the Gaussian Process

# %% [markdown]
# ### 3.1 The kernel
# The kernel says how similar two geometries are:
#
# $$k(\mathbf f, \mathbf f') = C \exp\!\Big[-\sum_d \frac{(f_d - f_d')^2}{2\ell_d^2}\Big] + \sigma_n^2\,\delta_{\mathbf f\mathbf f'}.$$
#
# Each feature has its own **length scale** $\ell_d$, learned from the data by maximising the likelihood. The white-noise term
# $\sigma_n^2$ absorbs the tiny numerical noise of the DFT energies. Features are standardised (mean 0, standard deviation 1) first
# ([Rasmussen & Williams, Ch. 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf);
# [Deringer et al. 2021](https://doi.org/10.1021/acs.chemrev.1c00022)).

# %% [markdown]
# ### <font color=blue> **Question 3.1.** </font>
# 1. In `fit_energy_gp`, complete the two lines that add the **swapped copy** of every training point: features with $r_1$ and $r_2$
#    exchanged, `F[:, [1, 0, 2]]`, and the same energies again.
# 2. Below it, compute `rms`, the RMS error of the GP on **your** test set, in **meV**.
#
# 🐍 **Python tip — `np.concatenate([a, b])`** joins two arrays end to end. `F[:, [1, 0, 2]]` picks the columns in the order 1, 0, 2.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `Fs = np.concatenate([F, F[:, [1, 0, 2]]])`, `Es = np.concatenate([E, E])`, and
# `rms = np.sqrt(np.mean((E_pred - E_test)**2)) * 1000`.
# </details>

# %%
def fit_energy_gp(F, E):
    """Fit a GP to features F (n x 3) and energies E (eV), including the O<->O swap. Returns (gp, mean, scale)."""
    Fs = np.concatenate([F, F[:, [1, 0, 2]]])  # SOLUTION
    Es = np.concatenate([E, E])  # SOLUTION
    mean, scale = Fs.mean(0), Fs.std(0)
    kernel = ConstantKernel(1.0, (1e-3, 1e7)) * RBF([1.0, 1.0, 1.0], (1e-2, 1e5)) + WhiteKernel(1e-5, (1e-6, 1e-2))
    gp = GaussianProcessRegressor(kernel, normalize_y=True).fit((Fs - mean) / scale, Es)
    return gp, mean, scale

t = time.time()
gp, f_mean, f_scale = fit_energy_gp(F_train, E_train)
print(f"fitted in {time.time() - t:.1f} s:", gp.kernel_)

E_pred, E_std = gp.predict((F_test - f_mean) / f_scale, return_std=True)
rms = np.sqrt(np.mean((E_pred - E_test)**2)) * 1000  # SOLUTION
print(f"test RMS error = {rms:.3f} meV  (= {rms / 1000 * lk.EV_TO_CM1:.2f} cm⁻¹);  mean GP σ = {E_std.mean() * 1000:.3f} meV")

plt.figure(figsize=(4, 4))
plt.plot(E_test, E_pred, ".", ms=8); plt.plot([0, 0.6], [0, 0.6], "k--")
plt.xlabel("DFT energy / eV"); plt.ylabel("GP energy / eV"); plt.title("held-out test set"); plt.show()

# %%
lab.check('3.1')

# %% [markdown]
# ## 4. How Much Data Is Enough? A Learning Curve
# **Just run the cell below** (about 10 s). It refits the GP on the first 20, 40, 80, 160 and 320 training points and plots the
# test RMS error against the number of points on log–log axes.

# %%
# Just run this cell: the learning curve
ns, errs = [20, 40, 80, 160, 320], []
for n in ns:
    g, mn, sc = fit_energy_gp(F_train[:n], E_train[:n])
    errs.append(np.sqrt(np.mean((g.predict((F_test - mn) / sc) - E_test)**2)) * 1000)
plt.loglog(ns, errs, "o-"); plt.xlabel("training points"); plt.ylabel("test RMS error / meV")
plt.grid(True, which="both", alpha=0.3); plt.title("learning curve"); plt.show()
print(dict(zip(ns, np.round(errs, 3))))

# %% [markdown]
# ## 5. Do Not Trust It Where There Is No Data
# **Just run the cell below** (about 5 s). It stretches one bond from 1.0 to 2.0 Å (the other fixed at $r_e$, linear) and plots
# the GP energy with its ±2σ band, together with DFT energies along the same cut.

# %%
# Just run this cell: a one-bond stretch, GP against DFT
r_cut = np.linspace(1.0, 2.0, 101)
F_cut = np.stack([r_cut, np.full_like(r_cut, X0[2, 2]), -np.ones_like(r_cut)], 1)
mu_cut, sd_cut = gp.predict((F_cut - f_mean) / f_scale, return_std=True)
r_chk = np.array([1.05, 1.25, 1.40, 1.55, 1.70, 1.85, 2.0])
E_chk = [co2_energy_dipole(np.array([[0, 0, -r], [0, 0, 0], X0[2]]))[0] - E_min for r in r_chk]
plt.fill_between(r_cut, mu_cut - 2 * sd_cut, mu_cut + 2 * sd_cut, alpha=0.3, label="GP ± 2σ")
plt.plot(r_cut, mu_cut, label="GP"); plt.plot(r_chk, E_chk, "ko", label="DFT")
plt.axvspan(F_train[:, :2].min(), F_train[:, :2].max(), color="0.9", zorder=-1, label="range of training data")
plt.xlabel("r₁ / Å"); plt.ylabel("E / eV"); plt.ylim(-0.5, 8); plt.legend(loc="upper left"); plt.show()

# %% [markdown]
# ### <font color=blue> **Question 5.1 Discussion.** </font>
# <font color=green>
#
# 1. From the learning curve: how does the error fall as the training set grows, and why does it level off?
# 2. Does the GP know how accurate it is? Compare its mean σ with the actual test RMS error.
# 3. Along the bond stretch, where does the GP fail, and does its uncertainty warn you? What does this mean for molecular dynamics
#    at a high temperature?
# </font>
#
# ---

# %% [markdown] answer
# The error falls steeply at first (from 0.2 meV with 20 points) and then levels off near 0.02 meV: once the surface is resolved, what remains is the numerical noise of the DFT integration grid, which no amount of data removes. The GP's mean σ (0.17 meV) is about eight times the actual test error (0.02 meV): it is cautious, so it overestimates its error here, but it is the right warning sign where data are missing. Along the bond stretch the GP follows DFT inside the training range (about 1.05–1.30 Å) and goes badly wrong outside it, but its ±2σ band widens there, warning that it is extrapolating. In MD at high temperature the molecule could wander outside the sampled region, where the forces would be meaningless; the sampling energy must cover the energies the dynamics will reach.

# %% [markdown]
# ---

# %%
lab.check('5.1d')

# %% [markdown]
# ## 6. Physical Check: Harmonic Frequencies of the GP Surface
# A small energy error can still hide a wrong *shape*. The curvature at the minimum is a sharper test.

# %% [markdown]
# ### <font color=blue> **Question 6.1.** </font>
# The cell builds the 9 × 9 Hessian of the GP surface by central finite differences (step 0.002 Å) and mass-weights it, as in
# Lab 4. Complete the line that converts the **last four** eigenvalues `lam[5:]` to wavenumbers `wn_gp` (cm⁻¹), as you did in Lab 4.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `wn_gp = lk.omega_to_wavenumber(np.sqrt(np.abs(lam[5:]) * lk.ACC))`.
# </details>

# %%
def gp_energy(X):
    """GP energy (eV) at one geometry (3, 3) or many (n, 3, 3)."""
    F = features(np.asarray(X)).reshape(-1, 3)
    return gp.predict((F - f_mean) / f_scale).reshape(np.shape(X)[:-2])

step, x0 = 0.002, X0.ravel()
Hgp = np.zeros((9, 9))
for i in range(9):
    for j in range(i, 9):
        P = []
        for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            x = x0.copy(); x[i] += si * step; x[j] += sj * step; P.append(x.reshape(3, 3))
        e = gp_energy(np.array(P))
        Hgp[i, j] = Hgp[j, i] = (e[0] - e[1] - e[2] + e[3]) / (4 * step**2)
lam = np.linalg.eigvalsh(Hgp / np.sqrt(np.outer(m3, m3)))
wn_gp = lk.omega_to_wavenumber(np.sqrt(np.abs(lam[5:]) * lk.ACC))  # SOLUTION
for a, b in zip(wn_gp, wn_dft):
    print(f"GP {a:7.1f}   DFT {b:7.1f}   difference {a - b:+5.1f} cm⁻¹")

# %%
lab.check('6.1')

# %% [markdown] instructor
# Reference run: GP − DFT = +0.1 (bend), −0.5 (ν₁) and +4.9 cm⁻¹ (ν₃). The ν₃ difference is not a fitting error: the GP was fitted to
# DFT *energies*, so its curvature matches the finite-difference DFT Hessian, which includes the motion of the integration grid. The
# analytic Hessian of Lab 4 leaves that term out and is 0.45 % softer along the carbon z coordinate (Question 2.2 of Lab 4), which is
# +5 cm⁻¹ at 2470 cm⁻¹. A good question for strong students: which of the two "DFT frequencies" is the right one?

# %% [markdown]
# ## 7. A Model for the Dipole Moment

# %% [markdown]
# ### 7.1 Bond charges
# The IR spectrum from MD needs the dipole at every step too. A simple, physical model: put a partial charge $q_1$ on O₁ and $q_2$
# on O₂ (and $-(q_1+q_2)$ on C). Then
#
# $$\boldsymbol\mu = q_1\,\mathbf b_1 + q_2\,\mathbf b_2 .$$
#
# Let each charge depend linearly on the geometry, $q_1 = \mathbf c\cdot\boldsymbol\phi(r_1, r_2, \cos\theta)$ with
# $\boldsymbol\phi = (1,\ r_1 - r_e,\ r_2 - r_e,\ 1 + \cos\theta)$, and $q_2$ the same with $r_1 \leftrightarrow r_2$ (symmetry again).
# Then $\boldsymbol\mu$ is *linear* in the 4 coefficients $\mathbf c$: $\boldsymbol\mu = A\,\mathbf c$, and $\mathbf c$ comes from ordinary
# **linear least squares**. `dipole_design(X)` builds the matrix $A$ (3 rows per geometry, in debye per e·Å).

# %% [markdown]
# ### <font color=blue> **Question 7.1.** </font>
# 1. Fit `c_dip` with `np.linalg.lstsq(A_train, mu_train.ravel(), rcond=None)`, which returns several things; the coefficients
#    are the first, `[0]`.
# 2. Compute `rms_mu`, the RMS length of the error vector on your test set: $\sqrt{\langle |\boldsymbol\mu_\text{pred} - \boldsymbol\mu_\text{DFT}|^2\rangle}$ (D).
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `c_dip = np.linalg.lstsq(A_train, mu_train.ravel(), rcond=None)[0]` and
# `rms_mu = np.sqrt(np.mean(np.sum((mu_pred - mu_test)**2, axis=1)))`.
# </details>

# %%
r_e = X0[2, 2]

def phi(F):
    return np.stack([np.ones(len(F)), F[:, 0] - r_e, F[:, 1] - r_e, 1 + F[:, 2]], 1)

def dipole_design(X):
    """Design matrix A (3n x 4) with mu = A @ c, mu in debye."""
    F = features(X)
    b1, b2 = X[:, 0] - X[:, 1], X[:, 2] - X[:, 1]
    P1, P2 = phi(F), phi(F[:, [1, 0, 2]])
    A = (P1[:, None, :] * b1[:, :, None] + P2[:, None, :] * b2[:, :, None]) * lk.DEBYE_PER_E_ANG
    return A.reshape(-1, 4)

A_train = dipole_design(X_train)
c_dip = np.linalg.lstsq(A_train, mu_train.ravel(), rcond=None)[0]  # SOLUTION
mu_pred = (dipole_design(X_test) @ c_dip).reshape(-1, 3)
rms_mu = np.sqrt(np.mean(np.sum((mu_pred - mu_test)**2, axis=1)))  # SOLUTION
print("charge coefficients c =", np.round(c_dip, 4))
print(f"test RMS dipole error {rms_mu:.4f} D  (RMS dipole {np.sqrt(np.mean(np.sum(mu_test**2, 1))):.3f} D)")

# %%
lab.check('7.1')

# %% [markdown]
# ### <font color=blue> **Question 7.2 Discussion.** </font>
# <font color=green>
#
# What is $q_1$ at the equilibrium geometry (hint: at $X_0$, $\boldsymbol\phi = (1, 0, 0, 0)$)? Is its sign what you expect from
# electronegativity? From the signs of $c_1$ and $c_2$: when bond 2 stretches and bond 1 shortens (the asymmetric stretch), where does
# the charge flow, and how does this make the asymmetric stretch so intense?
# </font>
#
# ---

# %% [markdown] answer
# At equilibrium q₁ = c₀ ≈ −0.27 e on each O (C ≈ +0.54 e), as electronegativity suggests. c₁ < 0 and c₂ > 0: when bond 2 stretches and bond 1 shortens, O₂ becomes more negative and O₁ less negative, so charge flows toward the oxygen that is moving away. This charge flux adds to the dipole change from the moving charges themselves, which is why ν₃ is so intense.

# %% [markdown]
# ---

# %%
lab.check('7.2d')

# %% [markdown]
# ## 8. Save the Surface for Lab 6
# **Just run the cell below.**

# %%
# Just run this cell: save the GP surface and dipole model for Lab 6
with open(lk.results_path("co2_ml_model.pkl"), "wb") as f:
    pickle.dump({"gp": gp, "f_mean": f_mean, "f_scale": f_scale, "c_dip": c_dip, "r_e": r_e,
                 "X0": X0, "symbols": symbols, "masses": masses, "wn_dft": wn_dft}, f)
print("saved results/co2_ml_model.pkl")

# %% [markdown]
# ### 📋 Summary of your results
# **Just run the cell below.**

# %%
# Just run this cell: summary table
summary = Table().with_columns(
    'Quantity', ['test RMS energy error (meV)', 'mean GP σ (meV)', 'learning curve: RMS with 20 points (meV)',
                 'GP − DFT, bend (cm⁻¹)', 'GP − DFT, symmetric stretch (cm⁻¹)', 'GP − DFT, asymmetric stretch (cm⁻¹)',
                 'q(O) at equilibrium (e)', 'test RMS dipole error (D)'],
    'Value', [round(rms, 3), round(E_std.mean() * 1000, 3), round(errs[0], 3), round(wn_gp[0] - wn_dft[0], 1),
              round(wn_gp[2] - wn_dft[2], 1), round(wn_gp[3] - wn_dft[3], 1), round(c_dip[0], 3), round(rms_mu, 4)])
summary.show()

# %% [markdown]
# ### 📝 Quick Check — Machine-Learned Surfaces
# Test what you've learned. This is just for practice — answer as many times as you like, it isn't recorded.

# %%
with open("questionsComp05.json", "r") as f:
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
# 1. How accurate the surface is, judged on held-out data and on the harmonic frequencies.
# 2. How the physics was built into the model (features, the O ↔ O swap, the bond-charge dipole).
# 3. Where the model cannot be trusted, and how you would know.
# 4. Whether the surface is good enough for MD at 300 K, and what you would change for 1500 K.
# </div>
#
# ---

# %% [markdown] answer
# On 100 held-out geometries the GP surface reproduces B3LYP to 0.02 meV RMS (0.2 cm⁻¹), close to the numerical noise of the DFT itself, and its harmonic frequencies agree with the DFT Hessian within 5 cm⁻¹, so both the energies and the shape of the surface near the minimum are right. The physics was built in rather than learned: the features (r₁, r₂, cos θ) are unchanged by translation and rotation, entering every point twice with r₁ ↔ r₂ makes the surface symmetric under the O ↔ O swap, and the dipole is written as bond charges whose four coefficients come from linear least squares (RMS error 0.01 D). The learning curve shows that 320 points is already past the point of diminishing returns. The model cannot be trusted outside the sampled region: along a single-bond stretch it fails beyond about 1.3 Å, and its growing σ is the warning. At 300 K the molecule holds about 0.1 eV of vibrational energy, well inside the 0.6 eV sampled, so the surface is adequate; for 1500 K I would sample to higher energy (about 1.5 eV) and check the test error there.

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
# (Instructor version: students write their reflection here.) The extrapolation plot was the most memorable result.

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
# 1. C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press (2006), [free online](https://gaussianprocess.org/gpml/), Ch. 2 (regression) and 5 (model selection).
# 2. V. L. Deringer, A. P. Bartók, N. Bernstein, D. M. Wilkins, M. Ceriotti and G. Csányi, "Gaussian process regression for materials and molecules," *Chem. Rev.* **121**, 10073 (2021), [doi:10.1021/acs.chemrev.1c00022](https://doi.org/10.1021/acs.chemrev.1c00022).
# 3. A. P. Bartók, M. C. Payne, R. Kondor and G. Csányi, "Gaussian approximation potentials," *Phys. Rev. Lett.* **104**, 136403 (2010), [doi:10.1103/PhysRevLett.104.136403](https://doi.org/10.1103/PhysRevLett.104.136403).
# 4. J. Behler, "Perspective: Machine learning potentials for atomistic simulations," *J. Chem. Phys.* **145**, 170901 (2016), [doi:10.1063/1.4966192](https://doi.org/10.1063/1.4966192).
# 5. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.* **12**, 2825 (2011).
# 6. Q. Sun et al., "Recent developments in the PySCF program package," *J. Chem. Phys.* **153**, 024109 (2020), [doi:10.1063/5.0006074](https://doi.org/10.1063/5.0006074).
# 7. P. Atkins, J. de Paula and J. Keeler, *Atkins' Physical Chemistry*, 11th ed., Oxford University Press (2018): Topic 11D (normal modes) and Focus 9 (computational chemistry).

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
lab.submit(name, lab_id='Comp05')
