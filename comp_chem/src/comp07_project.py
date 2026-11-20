# LAB
# %% [markdown]
# # Computational Lab 7 — Project: From Electrons to an Infrared Spectrum
# *CHEM 3376 — Physical Chemistry Laboratory: Data Analysis & Computation*
#
# **Project week.** Choose one triatomic molecule and carry out the full workflow you used for CO₂ in Labs 4–6:
#
# 1. the equilibrium geometry (B3LYP/def2-SVP);
# 2. harmonic normal modes and IR intensities;
# 3. a machine-learned surface and dipole model, tested on held-out data;
# 4. classical MD at 300 K and at 1000 K, with the 3-D movie;
# 5. the IR spectrum from MD, compared with the harmonic result and with experiment.
#
# You finish with **an abstract and one or two figures**, the format of a conference abstract.
#
# | molecule | atoms (outer–central–outer) | shape | experiment, cm⁻¹ (gas) | why it is interesting |
# |---|---|---|---|---|
# | N₂O | N–N–O | linear | ν₁ 1285, ν₂ 589, ν₃ 2224 | greenhouse gas; all modes IR-active, no inversion centre |
# | OCS | O–C–S | linear | ν₁ 859, ν₂ 520, ν₃ 2062 | the most abundant sulfur gas in the atmosphere |
# | HCN | H–C–N | linear | ν₁ 3311, ν₂ 712, ν₃ 2097 | a light atom: fast C–H stretch, needs a smaller time step |
# | CS₂ | S–C–S | linear | ν₁ 658 (Raman), ν₂ 397, ν₃ 1535 | same symmetry as CO₂; heavier atoms |
# | H₂O | H–O–H | bent | ν₁ 3657, ν₂ 1595, ν₃ 3756 | 3N − 6 = 3 modes, all IR-active; permanent dipole |
# | SO₂ | O–S–O | bent | ν₁ 1151, ν₂ 518, ν₃ 1362 | volcanic gas; bent, symmetric |
#
# Work through the sections in order. Most functions generalise ones you already wrote: the changes needed are listed in each
# section, and each check tests the **physics** (it works for any of the six molecules). Plan for about 1–2 hours of computer time
# in total; the slow cells say how long they take.
#
# <br>**<center>Learning Goals**
#
# |Area|Concept|
# |---|---|
# |Scientific practice|Planning and carrying out a complete computational study; judging every step against a physical test|
# |Vibrations|3N − 5 vs 3N − 6; symmetry and IR activity for linear, bent, symmetric and asymmetric triatomics|
# |Machine learning|When the permutation symmetry applies; generating your own training and test data|
# |Molecular dynamics|Choosing the time step for light atoms; temperature effects on bands|
# |Communication|An abstract and publication-quality figures|

# %% [markdown]
# ---
# ## 📄 Abstract
#
# <div style="border:2px solid #1f4e79; border-radius:8px; padding:10px 14px; background:#f3f7fb; color:#1a1a1a;">
#
# **Write this section last.** This abstract is the main product of the project. Your abstract goes in the cell below, replacing `ANSWER`.
#
# In **150–250 words**, one paragraph: the question; the methods (B3LYP/def2-SVP, harmonic analysis, Gaussian-process surface with
# its test error, classical MD); the main quantitative results, with experiment beside them; and the physical explanation of which
# vibrations absorb infrared light and why. A reader should understand what you found without your figures.
# </div>

# %% [markdown] answer
# The infrared spectrum of nitrous oxide (N₂O), an atmospheric greenhouse gas, was computed from first principles, from the harmonic approximation to classical molecular dynamics. The B3LYP/def2-SVP structure is linear, with r(N–N) = 1.130 Å and r(N–O) = 1.182 Å (experiment 1.128 and 1.184 Å). The harmonic analysis gives the doubly degenerate bend at 606 cm⁻¹ (2 × 12 km mol⁻¹), the N–O stretch at 1380 cm⁻¹ (50 km mol⁻¹) and the N–N stretch at 2418 cm⁻¹ (340 km mol⁻¹), against the observed ν₂ = 589, ν₁ = 1285 and ν₃ = 2224 cm⁻¹; all four modes are IR-active because N₂O has no inversion centre. A Gaussian-process surface trained on 200 DFT energies reproduces 50 held-out energies to 0.04 meV RMS and the harmonic frequencies to within 1 cm⁻¹, and an eight-parameter bond-charge model reproduces the held-out dipoles to 0.02 D. Classical MD with a 0.5 fs step conserves energy to 0.7 meV over 2 ps. Spectra averaged over 24 trajectories at 300 K place the bands at 607, 1380 and 2404 cm⁻¹, with rotational P–R contours, and the bands broaden and shift to lower wavenumber at 1000 K. The calculated stretches are 7–9 % above experiment, a known weakness of B3LYP for N₂O rather than an error of the dynamics, which reproduces the harmonic positions as classical MD should. Unlike CO₂, every vibration of N₂O absorbs infrared light. (Instructor version: N₂O reference run.)

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
ver = '2026Comp07V1'

# %% [markdown]
# ### 🧰 Standard imports

# %%
# Standard imports for this lab
import time
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyscf import gto, dft
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
for _up in ("..", os.path.join("..", "..")):          # labkit is in comp_chem/, the folder above this lab
    if os.path.isdir(os.path.join(_up, "labkit")):
        sys.path.insert(0, os.path.abspath(_up))
        break
import labkit as lk

# Answer checks for this lab: the checker (pchem_check.py) and the answer key (answers_comp07.json) are in this folder
from pchem_check import Checker
lab = Checker('comp07', 'answers_comp07.json')

# %% [markdown]
# Now run the cell below to enter your name.

# %%
# Enter your name as a string, e.g. name = 'Linus Pauling'
name = 'Instructor'  # SOLUTION

# %% [markdown]
# ## 0. Choose Your Molecule

# %% [markdown]
# ### <font color=blue> **Question 0.1.** </font>
# Set `SYSTEM` to the name of your molecule, as a string: `"N2O"`, `"OCS"`, `"HCN"`, `"CS2"`, `"H2O"` or `"SO2"`. The cell sets up the
# atoms (outer, central, outer), starting bond lengths and angle, whether the two outer atoms are identical (`symmetric`), and the
# MD time step (0.25 fs for molecules with hydrogen).

# %%
SYSTEMS = {   # symbols (outer, central, outer), starting r1, r2 (Å), angle (deg), identical outer atoms?, MD time step (fs)
    "N2O": dict(symbols=["N", "N", "O"], r1=1.13, r2=1.19, angle=180, symmetric=False, dt=0.5),
    "OCS": dict(symbols=["O", "C", "S"], r1=1.16, r2=1.56, angle=180, symmetric=False, dt=0.5),
    "HCN": dict(symbols=["H", "C", "N"], r1=1.07, r2=1.15, angle=180, symmetric=False, dt=0.25),
    "CS2": dict(symbols=["S", "C", "S"], r1=1.55, r2=1.55, angle=180, symmetric=True, dt=0.5),
    "H2O": dict(symbols=["H", "O", "H"], r1=0.96, r2=0.96, angle=104.5, symmetric=True, dt=0.25),
    "SO2": dict(symbols=["O", "S", "O"], r1=1.45, r2=1.45, angle=119, symmetric=True, dt=0.5),
}
EXPT = {"N2O": (1285, 589, 2224), "OCS": (859, 520, 2062), "HCN": (3311, 712, 2097),
        "CS2": (658, 397, 1535), "H2O": (3657, 1595, 3756), "SO2": (1151, 518, 1362)}   # ν₁, ν₂, ν₃ (cm⁻¹)

# choose your molecule, e.g. SYSTEM = "OCS"
SYSTEM = os.environ.get("LAB_SYSTEM", "N2O")  # SOLUTION
cfg = SYSTEMS[SYSTEM]
symbols, symmetric = cfg["symbols"], cfg["symmetric"]
masses = np.array([lk.MASS[s] for s in symbols])
m3 = np.repeat(masses, 3)
LINEAR = cfg["angle"] == 180
print(SYSTEM, symbols, "linear" if LINEAR else "bent", "symmetric" if symmetric else "asymmetric")

# %%
lab.check('0.1')

# %% [markdown]
# ## 1. Geometry
# **Changes from Lab 4:** the bonds need not be equal and the molecule may be bent, so a one-variable scan is not enough. Instead,
# minimise over all 9 Cartesian coordinates with `scipy.optimize.minimize(..., jac=True, method="BFGS")`, which needs the energy
# **and its gradient**. PySCF computes the gradient analytically: `mf.nuc_grad_method().kernel()`, in hartree/bohr, shape (3, 3).

# %% [markdown]
# ### <font color=blue> **Question 1.1.** </font>
# Complete `energy_and_gradient(x)`: convert the gradient to **eV/Å**, and return the energy (eV) and the gradient flattened to 9
# numbers (`.ravel()`). The optimisation takes 10 s to a minute. The check compares your gradient with a finite difference of the energy.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `g = mf.nuc_grad_method().kernel() * lk.HARTREE_TO_EV / lk.BOHR_TO_ANG` and `return mf.e_tot * lk.HARTREE_TO_EV, g.ravel()`.
# </details>

# %%
def build(r1, r2, angle_deg):
    """Outer atom 0 on −z, central atom at the origin, outer atom 2 in the xz plane."""
    a = np.radians(angle_deg)
    return np.array([[0, 0, -r1], [0, 0, 0], [r2 * np.sin(np.pi - a), 0, r2 * np.cos(np.pi - a)]])

def run_dft(X, basis="def2-svp", xc="b3lyp"):
    mol = gto.M(atom=[[s, tuple(x)] for s, x in zip(symbols, X)], basis=basis, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc; mf.kernel()
    return mf

def energy_and_gradient(x):
    """Energy (eV) and gradient (eV/Å, 9 numbers) for flattened coordinates x (Å)."""
    mf = run_dft(x.reshape(3, 3))
    g = mf.nuc_grad_method().kernel() * lk.HARTREE_TO_EV / lk.BOHR_TO_ANG  # SOLUTION
    return mf.e_tot * lk.HARTREE_TO_EV, g.ravel()  # SOLUTION

t = time.time()
res = minimize(energy_and_gradient, build(cfg["r1"], cfg["r2"], cfg["angle"]).ravel(), jac=True, method="BFGS",
               options={"gtol": 1e-4})
X0 = res.x.reshape(3, 3)
E_min = res.fun
b1, b2 = X0[0] - X0[1], X0[2] - X0[1]
angle0 = np.degrees(np.arccos(b1 @ b2 / np.linalg.norm(b1) / np.linalg.norm(b2)))
print(f"optimised in {time.time() - t:.0f} s ({res.nit} steps): r1 = {np.linalg.norm(b1):.4f} Å, r2 = {np.linalg.norm(b2):.4f} Å, "
      f"angle = {angle0:.2f}°")
lk.show_molecule(symbols, X0, labels=True)

# %%
lab.check('1.1')

# %% [markdown]
# ## 2. Harmonic Analysis and IR Intensities
# **Changes from Lab 4:** a bent molecule has 3N − 6 = 3 vibrations (6 near-zero eigenvalues), a linear one 3N − 5 = 4. The cell
# selects the vibrations as the eigenvalues above 100 cm⁻¹.

# %% [markdown]
# ### <font color=blue> **Question 2.1.** </font>
# Complete the two lines from Lab 4: the Hessian `H` in eV/Å² (9 × 9, rows ordered by atom then direction), and the intensity of
# mode `k` inside the loop (central difference of the dipole along the mode, step 0.01). The cell takes 10–60 s.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `H = mf0.Hessian().kernel().transpose(0, 2, 1, 3).reshape(9, 9) * lk.HARTREE_TO_EV / lk.BOHR_TO_ANG**2` and
# `A_k = 42.256 * np.sum(((dipole_dft(X0 + dX) - dipole_dft(X0 - dX)) / 0.02)**2)`.
# </details>

# %%
def dipole_dft(X):
    return run_dft(X).dip_moment(unit="Debye", verbose=0)

t = time.time()
mf0 = run_dft(X0)
H = mf0.Hessian().kernel().transpose(0, 2, 1, 3).reshape(9, 9) * lk.HARTREE_TO_EV / lk.BOHR_TO_ANG**2  # SOLUTION
lam, L = np.linalg.eigh(H / np.sqrt(np.outer(m3, m3)))
wn_all = lk.omega_to_wavenumber(np.sign(lam) * np.sqrt(np.abs(lam) * lk.ACC))
keep = wn_all > 100
wn, modes = wn_all[keep], L[:, keep]
disp = modes / np.sqrt(m3)[:, None]
n_modes = len(wn)
A = []
for k in range(n_modes):
    dX = (0.01 * disp[:, k]).reshape(3, 3)
    A_k = 42.256 * np.sum(((dipole_dft(X0 + dX) - dipole_dft(X0 - dX)) / 0.02)**2)  # SOLUTION
    A.append(A_k)
A = np.array(A)
print(f"({time.time() - t:.0f} s)  all eigenvalues (cm⁻¹): {np.round(wn_all, 0)}")
for k in range(n_modes):
    print(f"mode {k}: {wn[k]:7.1f} cm⁻¹   A = {A[k]:7.1f} km/mol")
print("dipole at equilibrium:", np.round(dipole_dft(X0), 3), "D;   experiment (ν₁, ν₂, ν₃):", EXPT[SYSTEM], "cm⁻¹")

# %%
lab.check('2.1')

# %% [markdown]
# **Just run the cell below** to watch each mode, and decide which experimental band (ν₁, ν₂, ν₃) it is.

# %%
# Just run this cell: animate each mode
for k in range(n_modes):
    print(f"mode {k}: {wn[k]:.0f} cm⁻¹")
    lk.animate_mode(symbols, X0, disp[:, k], amplitude=0.25)

# %% [markdown]
# ## 3. Training and Test Data
# **Changes from Lab 5:** `n_modes` modes instead of 4, and you compute your own training **and** test sets (there is no precomputed
# data for these molecules). Budget: about 0.3–1 s per point; 200 training and 50 test points take 1–5 minutes.

# %% [markdown]
# ### <font color=blue> **Question 3.1.** </font>
# Complete `sample_geometry`: the same two lines as Lab 5, with `n_modes` in place of 4. The check confirms that each geometry holds
# exactly the harmonic energy it was given.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `E_tot = E_max * rng.random()` and `Q = np.sqrt(2 * eps * lk.ACC) / omega * signs`.
# </details>

# %%
omega = 2 * np.pi * lk.C_CM_PER_FS * wn
N_TRAIN, N_TEST, E_MAX = 200, 50, 0.6

def sample_geometry(rng, E_max=E_MAX):
    E_tot = E_max * rng.random()  # SOLUTION
    eps = E_tot * rng.dirichlet(np.ones(n_modes))
    signs = rng.choice([-1, 1], n_modes)
    Q = np.sqrt(2 * eps * lk.ACC) / omega * signs  # SOLUTION
    return X0 + ((modes @ Q) / np.sqrt(m3)).reshape(3, 3)

print(np.round(sample_geometry(np.random.default_rng(0)), 3))

# %%
lab.check('3.1')

# %% [markdown]
# **Just run the cell below** (1–5 minutes). It computes the DFT energies and dipoles of 250 sampled geometries, saves them to
# `comp_chem/results/`, and splits them into 200 for training and 50 for testing.

# %%
# Just run this cell: your own DFT data
rng = np.random.default_rng(1)
t = time.time()
X_all = np.array([sample_geometry(rng) for _ in range(N_TRAIN + N_TEST)])
E_all, mu_all = [], []
for X in X_all:
    mf = run_dft(X)
    E_all.append(mf.e_tot * lk.HARTREE_TO_EV - E_min)
    mu_all.append(mf.dip_moment(unit="Debye", verbose=0))
E_all, mu_all = np.array(E_all), np.array(mu_all)
np.savez(lk.results_path(f"{SYSTEM}_samples.npz"), X=X_all, E=E_all, mu=mu_all, X0=X0)
print(f"{len(X_all)} points in {time.time() - t:.0f} s; saved results/{SYSTEM}_samples.npz")
X_tr, E_tr, mu_tr = X_all[:N_TRAIN], E_all[:N_TRAIN], mu_all[:N_TRAIN]
X_te, E_te, mu_te = X_all[N_TRAIN:], E_all[N_TRAIN:], mu_all[N_TRAIN:]

# %% [markdown]
# ## 4. The Machine-Learned Surface and Dipole
# **Changes from Lab 5:**
# - The features are the same, $(r_1, r_2, \cos\theta)$, with the central atom as atom 1.
# - Add the swapped copies **only if the two outer atoms are identical** (`symmetric`).
# - Dipole model: if the outer atoms differ, they need their own charge coefficients (8 instead of 4), with
#   $\boldsymbol\phi = (1,\ r_1 - r_{1e},\ r_2 - r_{2e},\ \cos\theta - \cos\theta_e)$.

# %% [markdown]
# ### <font color=blue> **Question 4.1.** </font>
# Complete the two lines that build the GP training set `Fs`, `Es`: with the swapped copies if `symmetric`, the plain data if not.
# Then complete the line in `dipole_design` that combines the two outer atoms' terms: **added** (one shared set of 4 coefficients) if
# `symmetric`, **side by side** (8 coefficients) if not. The GP fit takes 5–30 s.
#
# 🐍 **Python tip — a one-line `if`.** `a if condition else b` is `a` when the condition is true and `b` otherwise.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `Fs = np.concatenate([F_tr, F_tr[:, [1, 0, 2]]]) if symmetric else F_tr`, the same for `Es` with `[E_tr, E_tr]`, and
# `A = A1 + A2 if symmetric else np.concatenate([A1, A2], -1)`.
# </details>

# %%
def features(X):
    b1, b2 = X[..., 0, :] - X[..., 1, :], X[..., 2, :] - X[..., 1, :]
    r1, r2 = np.linalg.norm(b1, axis=-1), np.linalg.norm(b2, axis=-1)
    return np.stack([r1, r2, np.sum(b1 * b2, -1) / (r1 * r2)], -1)

F_tr, F_te = features(X_tr), features(X_te)
Fs = np.concatenate([F_tr, F_tr[:, [1, 0, 2]]]) if symmetric else F_tr  # SOLUTION
Es = np.concatenate([E_tr, E_tr]) if symmetric else E_tr  # SOLUTION
f_mean, f_scale = Fs.mean(0), Fs.std(0)
kernel = ConstantKernel(1.0, (1e-3, 1e7)) * RBF([1.0] * 3, (1e-2, 1e5)) + WhiteKernel(1e-5, (1e-6, 1e-2))
t = time.time()
gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2).fit((Fs - f_mean) / f_scale, Es)
print(f"fitted in {time.time() - t:.0f} s:", gp.kernel_)

def energy(X):
    F = features(np.asarray(X)).reshape(-1, 3)
    return gp.predict((F - f_mean) / f_scale).reshape(np.shape(X)[:-2])

rms = np.sqrt(np.mean((energy(X_te) - E_te)**2)) * 1000
print(f"energy test RMS = {rms:.3f} meV = {rms / 1000 * lk.EV_TO_CM1:.2f} cm⁻¹")

Fe = features(X0)
def phi(F, swap=False):
    F = F[..., [1, 0, 2]] if swap else F
    r1e, r2e = (Fe[1], Fe[0]) if swap else (Fe[0], Fe[1])
    return np.stack([np.ones(F.shape[:-1]), F[..., 0] - r1e, F[..., 1] - r2e, F[..., 2] - Fe[2]], -1)

def dipole_design(X):
    F = features(X)
    b1, b2 = X[..., 0, :] - X[..., 1, :], X[..., 2, :] - X[..., 1, :]
    A1 = phi(F)[..., None, :] * b1[..., :, None]
    A2 = phi(F, swap=True)[..., None, :] * b2[..., :, None]
    A = A1 + A2 if symmetric else np.concatenate([A1, A2], -1)  # SOLUTION
    return lk.DEBYE_PER_E_ANG * A

c_dip = np.linalg.lstsq(dipole_design(X_tr).reshape(-1, 4 if symmetric else 8), mu_tr.ravel(), rcond=None)[0]

def dipole(X):
    return dipole_design(np.asarray(X)) @ c_dip

rms_mu = np.sqrt(np.mean(np.sum((dipole(X_te) - mu_te)**2, -1)))
print(f"dipole test RMS = {rms_mu:.4f} D;  RMS dipole {np.sqrt(np.mean(np.sum(mu_te**2, -1))):.3f} D")

# %%
lab.check('4.1')

# %% [markdown]
# **Just run the cell below.** The harmonic frequencies of the GP surface (the physical check of Lab 5) and the parity plot, a
# candidate for Figure 2.

# %%
# Just run this cell: harmonic check of the GP surface, and the parity plot
step, x0 = 0.002, X0.ravel()
Hgp = np.zeros((9, 9))
for i in range(9):
    for j in range(i, 9):
        P = []
        for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            x = x0.copy(); x[i] += si * step; x[j] += sj * step; P.append(x.reshape(3, 3))
        e = energy(np.array(P)); Hgp[i, j] = Hgp[j, i] = (e[0] - e[1] - e[2] + e[3]) / (4 * step**2)
lam_gp = np.linalg.eigvalsh(Hgp / np.sqrt(np.outer(m3, m3)))
wn_gp = lk.omega_to_wavenumber(np.sqrt(np.abs(lam_gp) * lk.ACC))[-n_modes:]
for a, b in zip(wn_gp, wn):
    print(f"GP {a:7.1f}  DFT {b:7.1f}  ({a - b:+.1f} cm⁻¹)")
plt.figure(figsize=(3.5, 3.5)); plt.plot(E_te, energy(X_te), "."); plt.plot([0, E_te.max()], [0, E_te.max()], "k--")
plt.xlabel("DFT / eV"); plt.ylabel("ML / eV"); plt.title(f"{SYSTEM}: held-out test"); plt.show()

# %% [markdown]
# ## 5. Molecular Dynamics
# **Changes from Lab 6:** use `cfg["dt"]` (0.25 fs for molecules with H), `n_modes` modes in the thermal start, and the new `energy`
# and `dipole` functions.

# %% [markdown]
# ### <font color=blue> **Question 5.1.** </font>
# Complete `forces` and the three update lines of `run_md`, copied from Lab 6 (`masses_c` here is the masses as a column). The cell
# runs 2 ps at 300 K (10 s to 2 minutes) and shows the movie. The check tests energy conservation.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `return -((e[0::2] - e[1::2]) / (2 * h)).reshape(3, 3)`; `V += 0.5 * dt * F / masses_c * lk.ACC` (twice) and `X += dt * V`.
# </details>

# %%
masses_c = masses[:, None]

def forces(X, h=1e-3):
    P = np.repeat(X[None], 18, axis=0)
    for k in range(9):
        P[2 * k].flat[k] += h; P[2 * k + 1].flat[k] -= h
    e = energy(P)
    return -((e[0::2] - e[1::2]) / (2 * h)).reshape(3, 3)  # SOLUTION

def run_md(X, V, dt, n_steps, save_every=1):
    X, V = X.copy(), V.copy(); traj, mus, et = [], [], []
    F = forces(X)
    for step in range(n_steps):
        V += 0.5 * dt * F / masses_c * lk.ACC  # SOLUTION
        X += dt * V  # SOLUTION
        F = forces(X)
        V += 0.5 * dt * F / masses_c * lk.ACC  # SOLUTION
        if step % save_every == 0:
            traj.append(X.copy()); mus.append(dipole(X))
            et.append(energy(X) + 0.5 * np.sum(masses_c * V**2) / lk.ACC)
    return np.array(traj), np.array(mus), np.array(et)

def thermal_start(T, rng):
    kT = lk.KB_EV * T
    Q = rng.normal(size=n_modes) * np.sqrt(kT * lk.ACC) / omega
    X = X0 + ((modes @ Q) / np.sqrt(m3)).reshape(3, 3)
    V = rng.normal(size=(3, 3)) * np.sqrt(kT * lk.ACC / masses_c)
    V -= (masses_c * V).sum(0) / masses_c.sum()
    return X, V

def ir_spectrum(mu, dt_fs):
    m = (mu - mu.mean(0)) * np.hanning(len(mu))[:, None]
    n = 4 * len(m)
    nu = np.fft.rfftfreq(n, d=dt_fs) / lk.C_CM_PER_FS
    return nu, np.sum(np.abs(np.fft.rfft(m, n=n, axis=0))**2, 1) * nu**2

dt = cfg["dt"]
X, V = thermal_start(300, np.random.default_rng(5))
t = time.time()
traj, mu, et = run_md(X, V, dt, int(2000 / dt), save_every=1)
print(f"2 ps in {time.time() - t:.0f} s; energy fluctuation {1000 * np.ptp(et):.2f} meV")
lk.animate_trajectory(symbols, traj, stride=max(1, int(1 / dt)), interval=30)

# %%
lab.check('5.1')

# %% [markdown]
# **Just run the cell below** (2–10 minutes). It averages the spectra of 24 trajectories (4 ps each) at 300 K and at 1000 K.

# %%
# Just run this cell: averaged spectra at two temperatures
def averaged_spectrum(T, n_traj=24, t_ps=4, seed=0):
    rng, acc = np.random.default_rng(seed), 0
    every = max(1, int(1 / dt))
    for _ in range(n_traj):
        X, V = thermal_start(T, rng)
        _, m, _ = run_md(X, V, dt, int(t_ps * 1000 / dt), save_every=every)
        nu, I = ir_spectrum(m, every * dt)
        acc = acc + I
    return nu, acc / n_traj

t = time.time()
spectra = {T: averaged_spectrum(T) for T in (300, 1000)}
print(f"({time.time() - t:.0f} s)")
nu300, I300 = spectra[300]
md_peaks = []
for k in range(n_modes):
    w = (nu300 > wn[k] - 80) & (nu300 < wn[k] + 80)
    md_peaks.append(nu300[w][np.argmax(I300[w])])
print("MD 300 K peak near each harmonic mode (cm⁻¹):", np.round(md_peaks, 0))

# %% [markdown]
# ## 6. Figures for Your Report
# A good figure tells one story. Suggested Figure 1 (below): the MD spectra at two temperatures with the harmonic sticks and the
# experimental fundamentals marked. Suggested Figure 2: the molecule and its normal modes, a film strip of the trajectory, or the ML
# parity plot. Label axes with units. **Just run the two cells below**; they save PNG files at 300 dpi in this folder (download them
# from the file browser). Change the plots if you like: they are your figures.

# %%
# Figure 1: MD spectra, harmonic sticks and experiment
fig, ax = plt.subplots(figsize=(7, 3.6))
top = max(wn.max(), max(EXPT[SYSTEM])) + 300
for (T, (nu_T, I_T)), col in zip(spectra.items(), ("#1f4e79", "#c0504d")):
    w = (nu_T > 200) & (nu_T < top)
    ax.plot(nu_T[w], I_T[w] / I_T[w].max(), color=col, label=f"MD on ML surface, {T} K")
ax.vlines(wn, 0, A / A.max(), color="k", lw=2, label="harmonic B3LYP")
for e in EXPT[SYSTEM]:
    ax.axvline(e, color="0.5", ls=":", lw=1)
ax.plot([], [], color="0.5", ls=":", label="experiment")
ax.set_xlabel("wavenumber / cm⁻¹"); ax.set_ylabel("IR absorbance (normalised)"); ax.set_title(SYSTEM)
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{SYSTEM}_figure1.png", dpi=300); plt.show()

# %%
# Figure 2 candidate: a film strip of the trajectory
fig, ax = plt.subplots(figsize=(9, 1.8))
lk.trajectory_strip(symbols, traj, np.arange(len(traj)) * dt, frames=list(range(0, int(30 / dt), int(5 / dt))), ax=ax)
plt.savefig(f"{SYSTEM}_trajectory_strip.png", dpi=300, bbox_inches="tight"); plt.show()

# %%
lab.check('6.1')

# %% [markdown]
# ## 7. Results

# %% [markdown]
# ### <font color=blue> **Question 7.1 Results table.** </font>
# <font color=green>
#
# Fill in this table for your molecule in the cell below (copy the table into the cell, replacing `ANSWER`, and fill in each row).
# Assign each harmonic mode to ν₁, ν₂ or ν₃ from the animations.
#
# | mode | description | harmonic B3LYP / cm⁻¹ | IR intensity / km mol⁻¹ | MD 300 K peak / cm⁻¹ | experiment / cm⁻¹ |
# |---|---|---|---|---|---|
# | | | | | | |
# </font>
#
# ---

# %% [markdown] answer
# | mode | description | harmonic B3LYP / cm⁻¹ | IR intensity / km mol⁻¹ | MD 300 K peak / cm⁻¹ | experiment / cm⁻¹ |
# |---|---|---|---|---|---|
# | ν₂ (×2) | bend | 606 | 2 × 12 | 607 | 589 |
# | ν₁ | N–O stretch (mostly) | 1380 | 50 | 1380 | 1285 |
# | ν₃ | N–N stretch (mostly) | 2418 | 340 | 2404 | 2224 |
#
# (Instructor version: N₂O reference run.)

# %% [markdown]
# ---

# %%
lab.check('7.1t')

# %% [markdown]
# ### <font color=blue> **Question 7.2 Discussion.** </font>
# <font color=green>
#
# 1. How many modes does your molecule have, and why? Which are IR-active, and why (symmetry)? Compare with CO₂.
# 2. How do the harmonic, MD and experimental band positions compare? Explain the differences (functional and basis set,
#    anharmonicity, classical vs quantum).
# 3. What changes between 300 K and 1000 K? If your molecule has a permanent dipole, what is the band at low wavenumber
#    (below 200 cm⁻¹, cut off in Figure 1)?
# </font>
#
# ---

# %% [markdown] answer
# N₂O is linear, so it has 3N − 5 = 4 modes: the doubly degenerate bend and two stretches. Unlike CO₂ it has no inversion centre, so all its modes change the dipole and all are IR-active, including ν₁ (the mostly N–O stretch), which in CO₂ is dark. Harmonic B3LYP/def2-SVP is 3 % high for the bend and 7–9 % high for the stretches: B3LYP is known to do poorly for N₂O, whose bonding mixes N=N=O and N≡N–O structures (it also gives a dipole of only 0.006 D against the observed 0.16 D). Classical MD at 300 K reproduces the harmonic positions within about 15 cm⁻¹, because it has no zero-point energy and stays near the bottom of the well; most of the difference from experiment is the functional. At 1000 K the bands broaden, shift to lower wavenumber (anharmonicity), and the P–R contours spread. A polar molecule also has a pure-rotation band at low wavenumber, but the B3LYP dipole of N₂O is so small that it is very weak here.

# %% [markdown]
# ---

# %%
lab.check('7.2d')

# %% [markdown]
# ### 📝 Quick Check — The Full Workflow
# Test what you've learned. This is just for practice — answer as many times as you like, it isn't recorded.

# %%
with open("questionsComp07.json", "r") as f:
    display_quiz(json.load(f), shuffle_answers=True)

# %% [markdown]
# ---
# ## 8. Conclusions

# %% [markdown]
# ### <font color=blue> **Question 8. Conclusions.** </font>
# <div style="border:2px solid #1f4e79; border-radius:8px; padding:10px 14px; background:#f3f7fb; color:#1a1a1a;">
#
# Write the **Conclusions** of your project in the cell below (one or two paragraphs, about 150–300 words). Address:
#
# 1. The main result: which vibrations of your molecule absorb infrared light, where, and why.
# 2. How you checked each step (gradient, mode count, test error, energy conservation) and which step limits the accuracy most.
# 3. How your molecule differs from CO₂, and what that means for its role in the atmosphere (or wherever it matters).
# 4. What you would do with a week more of computer time.
# </div>
#
# ---

# %% [markdown] answer
# All four vibrations of N₂O absorb infrared light: with no inversion centre, every mode changes the dipole. The strongest band is the N–N stretch (ν₃, observed 2224 cm⁻¹), followed by the N–O stretch (1285 cm⁻¹) and the bend (589 cm⁻¹). Each step was checked: the analytic gradient matched a finite difference of the energy, the Hessian gave the four vibrations of a linear molecule, the GP reproduced held-out DFT energies to 0.04 meV and its own harmonic frequencies to 1 cm⁻¹, and MD conserved energy to 0.7 meV. The accuracy is limited by B3LYP/def2-SVP, which puts the stretches 7–9 % too high and the dipole an order of magnitude too low, and by classical MD, which has no zero-point energy. Unlike CO₂, N₂O absorbs at 1285 cm⁻¹ (7.8 µm), at the edge of the 8–13 µm atmospheric window where little else absorbs, which helps make it about 270 times more potent than CO₂ per molecule over a century. With more computer time I would rebuild the surface with CCSD(T) (or at least a larger basis and a different functional), and include zero-point energy with quantum methods (vibrational SCF or path-integral MD).

# %% [markdown]
# ---

# %% [markdown]
# ## 9. Reflection

# %% [markdown]
# ### <font color=blue> **Question 9.** </font>
# At the end of the project, please include a reflection.
# * How did the project go? Which step was hardest to generalise from CO₂?
# * Which part of the course's Python was most useful?
# * Did you use the hints? Which questions needed them?
# * What would you like your instructor to review in class?
# * How long did the project take you to complete?
#
# **Use the markdown cell below this one and write your reflection on this project.**
#
# ---

# %% [markdown] answer
# (Instructor version: students write their reflection here.) The project tied the six labs together.

# %% [markdown]
# ---

# %% [markdown]
# #### Save notebook before check
# This cell checks your **Abstract** (at the top of the notebook), your **Conclusions** and your **Reflection**.

# %%
lab.check('abstract')
lab.check('8c')
lab.check('9d')

# %% [markdown] instructor
# What to look for in student work: (1) the correct mode count (4 for linear, 3 for bent) and an explanation of why all modes of
# N₂O/OCS/HCN are IR-active (no inversion centre) while ν₁ of CS₂ is not; (2) a test-set error, not a training error; (3) a time step
# suited to the fastest mode (0.25 fs with H); (4) an honest comparison: harmonic B3LYP overestimates stretches by a few per cent, and
# classical MD at 300 K gives near-harmonic band positions, not quantum fundamentals (no zero-point energy); (5) for molecules with a
# permanent dipole (N₂O, OCS, HCN, H₂O, SO₂) the MD spectrum has a pure-rotation band at low wavenumber, which students may notice
# and should be able to explain (Figure 1 starts at 200 cm⁻¹). H₂O: the bend overtone and the near-degenerate stretches make a good
# discussion; SO₂ and CS₂ are the slowest (heavier atoms, more electrons). The reference run in this notebook is N₂O; set the
# environment variable LAB_SYSTEM (e.g. LAB_SYSTEM=H2O) before building with --execute to run another molecule.

# %% [markdown]
# ## 📚 References
# 1. Q. Sun et al., "Recent developments in the PySCF program package," *J. Chem. Phys.* **153**, 024109 (2020), [doi:10.1063/5.0006074](https://doi.org/10.1063/5.0006074).
# 2. P. Atkins, J. de Paula and J. Keeler, *Atkins' Physical Chemistry*, 11th ed., Oxford University Press (2018): Topics 11D and 11E (vibrations of polyatomic molecules and their symmetry analysis).
# 3. T. Shimanouchi, *Tables of Molecular Vibrational Frequencies, Consolidated Volume I*, NSRDS-NBS 39 (1972); values via the [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/).
# 4. V. L. Deringer et al., "Gaussian process regression for materials and molecules," *Chem. Rev.* **121**, 10073 (2021), [doi:10.1021/acs.chemrev.1c00022](https://doi.org/10.1021/acs.chemrev.1c00022).
# 5. D. Frenkel and B. Smit, *Understanding Molecular Simulation*, 2nd ed., Academic Press (2002).
# 6. IPCC, *Climate Change 2021: The Physical Science Basis*, Ch. 7 (global warming potentials), Cambridge University Press (2021), [doi:10.1017/9781009157896](https://doi.org/10.1017/9781009157896).

# %% [markdown]
# ## 🏁 Final Step — Check, Submit, Then Upload to Canvas
#
# Follow these steps **in order**:
#
# 1. 💾 **Save** the notebook (Ctrl/Cmd + S): the written-answer checks and the submission read the *saved* file.
# 2. ✅ **Run the cell below** to see your score: one row per question, with the number of attempts. Any question still marked ❌ or
#    *not checked*: go back, fix it, and run its check again.
# 3. 📨 **Click Submit Lab.** This records your score, your abstract, your results table, your conclusions and your discussion under
#    your JupyterHub username. You can submit again after making changes; the latest submission counts.
# 4. 📤 **Make a PDF and upload it to Canvas,** together with your figure PNG files. Export an HTML copy ([instructions](https://laserchemist.github.io/data/submission-instructions/export-html)), open it in your browser, and use **Print → Save as PDF**. Check that your plots, tables and written answers all appear in the PDF.

# %%
# Your score for this lab, and the Submit button
lab.score()
lab.submit(name, lab_id='Comp07')
