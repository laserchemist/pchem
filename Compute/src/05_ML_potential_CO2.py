# %% [markdown]
# # Notebook 5 — A machine-learned potential energy surface for CO₂
#
# **Session 2, about 60 minutes**
#
# The harmonic picture of Notebook 4 is exact only for infinitesimal vibrations. To see how a real,
# warm molecule moves (anharmonicity, rotation, energy flowing between modes) we will run **molecular
# dynamics** in Notebook 6. MD needs the energy and forces at every time step: 20 000 steps × 0.7 s per DFT
# calculation is about 4 hours. Instead we will **learn** the surface from a few hundred DFT calculations, and
# evaluate it in microseconds.
#
# The method is **Gaussian process regression (GPR)**: it predicts the energy at a new geometry as a smooth,
# weighted combination of the energies at the training geometries, weighted by how *similar* the geometries are.
# It also gives an **uncertainty**, which tells you when it is guessing.
#
# Four principles carry over from research-grade machine-learned surfaces:
#
# 1. **Build the physics into the inputs.** The energy cannot change if the molecule is translated or rotated, or if
#    the two identical O atoms are swapped. Use coordinates that have those symmetries.
# 2. **Sample where the dynamics will go**, a little beyond, and not much further.
# 3. **Judge the fit on data it has never seen** (a held-out test set), not on its training error.
# 4. **Check physical quantities**, such as the harmonic frequencies, against the reference calculation.

# %%
import sys, os, time, pickle
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, dft
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import labkit as lk

fname = "co2_harmonic.npz" if os.path.exists("co2_harmonic.npz") else lk.data_path("co2_harmonic.npz")
h = np.load(fname)
symbols = [str(s) for s in h["symbols"]]
X0, wn_dft, modes, masses = h["X0"], h["wn"], h["modes"], h["masses"]
m3 = np.repeat(masses, 3)
print("harmonic DFT wavenumbers (cm^-1):", np.round(wn_dft, 1))

def co2_energy_dipole(X, basis="def2-svp", xc="b3lyp"):
    mol = gto.M(atom=[[s, tuple(x)] for s, x in zip(symbols, X)], basis=basis, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc
    e = mf.kernel() * lk.HARTREE_TO_EV
    return e, mf.dip_moment(unit="Debye", verbose=0)

E_min, _ = co2_energy_dipole(X0)

# %% [markdown]
# ## 1. Sampling geometries
# A good way to generate realistic distorted geometries is to displace along the **normal modes**. If mode $k$
# (mass-weighted coordinate $Q_k$, frequency $\omega_k$) holds energy $\varepsilon_k$, its harmonic amplitude is
# $Q_k = \sqrt{2\varepsilon_k}/\omega_k$. The Cartesian displacement is $\Delta\mathbf x = \sum_k Q_k \mathbf L_k/\sqrt{m}$.
#
# **Exercise 1.** Write `sample_geometry(rng, E_max)` that
# - draws a total energy between 0 and `E_max` (eV),
# - splits it randomly among the 4 modes (`rng.dirichlet(np.ones(4))`),
# - gives each amplitude a random sign, and returns the displaced geometry.
#
# $\omega_k$ in rad/fs comes from the wavenumber: $\omega = 2\pi c\tilde\nu$. Remember the energy units:
# $\tfrac12\omega^2 Q^2$ with $Q$ in Å·amu$^{1/2}$ and $\omega$ in rad/fs is in amu·Å²/fs², which is $1/\texttt{ACC}$ eV.

# %%
omega = 2 * np.pi * lk.C_CM_PER_FS * wn_dft          # rad/fs

def sample_geometry(rng, E_max=0.6):
    """Random geometry with up to E_max (eV) of harmonic energy spread over the normal modes."""
    ### BEGIN SOLUTION
    E_tot = E_max * rng.random()
    eps = E_tot * rng.dirichlet(np.ones(4))
    Q = np.sqrt(2 * eps * lk.ACC) / omega * rng.choice([-1, 1], 4)
    dx = (modes @ Q) / np.sqrt(m3)
    return X0 + dx.reshape(3, 3)
    ### END SOLUTION

rng = np.random.default_rng(2024)
print(np.round(sample_geometry(rng), 3))

# %% [markdown]
# **Exercise 2.** Compute B3LYP energies (relative to `E_min`) and dipoles for **100** of your own geometries
# (about 1–2 minutes). These are your **test set**. For training we will use 320 geometries computed the same way,
# which are provided in `data/` so that everyone has the same training set.

# %%
N_TEST = 100
t = time.time()
### BEGIN SOLUTION
X_test = np.array([sample_geometry(rng) for _ in range(N_TEST)])
ED = [co2_energy_dipole(X) for X in X_test]
E_test = np.array([e for e, _ in ED]) - E_min
mu_test = np.array([m for _, m in ED])
### END SOLUTION
print(f"{N_TEST} DFT points in {time.time() - t:.0f} s;  energies {E_test.min():.3f} to {E_test.max():.3f} eV")

train = lk.load_co2_samples()
X_train, E_train, mu_train = train["X"], train["dE_eV"], train["dipole_D"]
print("training set:", X_train.shape[0], "points,", train["method"])

# %% [markdown]
# ## 2. Symmetry-adapted features
# **Exercise 3.** Write `features(X)` that maps geometries of shape `(..., 3, 3)` (atoms O, C, O) to
# $(r_1, r_2, \cos\theta)$: the two C–O distances and the cosine of the O–C–O angle.
#
# - Why is $\cos\theta$ better than $\theta$? (Hint: the linear molecule has θ = 180°, where $\theta$ has a kink
#   when the molecule bends through linear in any direction.)
# - $(r_1, r_2, \cos\theta)$ is unchanged by translation and rotation. To include the O↔O swap, train on every
#   geometry **twice**, once as $(r_1, r_2)$ and once as $(r_2, r_1)$.

# %%
def features(X):
    """(r1, r2, cos theta) for CO2 geometries X of shape (..., 3, 3), atom order O, C, O."""
    ### BEGIN SOLUTION
    b1 = X[..., 0, :] - X[..., 1, :]
    b2 = X[..., 2, :] - X[..., 1, :]
    r1 = np.linalg.norm(b1, axis=-1)
    r2 = np.linalg.norm(b2, axis=-1)
    cos = np.sum(b1 * b2, axis=-1) / (r1 * r2)
    return np.stack([r1, r2, cos], axis=-1)
    ### END SOLUTION

F_train, F_test = features(X_train), features(X_test)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(F_train[:, 0], F_train[:, 1], ".", label="train"); ax[0].plot(F_test[:, 0], F_test[:, 1], "x", label="your test")
ax[0].set_xlabel("r₁ / Å"); ax[0].set_ylabel("r₂ / Å"); ax[0].legend()
ax[1].hist(np.degrees(np.arccos(F_train[:, 2])), 30); ax[1].set_xlabel("O–C–O angle / deg")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Fitting the Gaussian process
# The kernel says how similar two geometries are:
#
# $$k(\mathbf f, \mathbf f') = C \exp\!\Big[-\sum_d \frac{(f_d - f_d')^2}{2\ell_d^2}\Big] + \sigma_n^2\,\delta_{\mathbf f\mathbf f'}.$$
#
# Each feature has its own length scale $\ell_d$, which is learned from the data by maximising the likelihood.
# The white-noise term $\sigma_n^2$ absorbs the tiny numerical noise of the DFT energies. Its lower bound
# (10⁻⁶ of the energy variance) keeps the fit numerically well conditioned, which matters for smooth forces in Notebook 6. Features are standardised
# (mean 0, standard deviation 1) first.
#
# **Exercise 4.** Fit the GP on the symmetrised training set and report the RMS error on **your** test set in meV.
# Also print the GP's own uncertainty (`return_std=True`). Does the GP know how accurate it is?

# %%
def fit_energy_gp(F, E):
    """Fit a GP to features F (n x 3) and energies E (eV), including the O<->O swap. Returns (gp, mean, scale)."""
    ### BEGIN SOLUTION
    Fs = np.concatenate([F, F[:, [1, 0, 2]]])
    Es = np.concatenate([E, E])
    ### END SOLUTION
    mean, scale = Fs.mean(0), Fs.std(0)
    kernel = ConstantKernel(1.0, (1e-3, 1e7)) * RBF([1.0, 1.0, 1.0], (1e-2, 1e5)) + WhiteKernel(1e-5, (1e-6, 1e-2))
    gp = GaussianProcessRegressor(kernel, normalize_y=True).fit((Fs - mean) / scale, Es)
    return gp, mean, scale

t = time.time()
gp, f_mean, f_scale = fit_energy_gp(F_train, E_train)
print(f"fitted in {time.time() - t:.1f} s:", gp.kernel_)

### BEGIN SOLUTION
E_pred, E_std = gp.predict((F_test - f_mean) / f_scale, return_std=True)
rms = np.sqrt(np.mean((E_pred - E_test)**2)) * 1000
### END SOLUTION
print(f"test RMS error = {rms:.3f} meV  (= {rms / 1000 * lk.EV_TO_CM1:.2f} cm^-1);  mean GP std = {E_std.mean() * 1000:.3f} meV")

plt.figure(figsize=(4, 4))
plt.plot(E_test, E_pred, ".", ms=8); plt.plot([0, 0.6], [0, 0.6], "k--")
plt.xlabel("DFT energy / eV"); plt.ylabel("GP energy / eV"); plt.title("held-out test set"); plt.show()

# %% [markdown]
# ## 4. How much data is enough? A learning curve
# **Exercise 5.** Fit the GP on the first 20, 40, 80, 160 and 320 training points and plot the test RMS error
# against the number of points on log–log axes.

# %%
### BEGIN SOLUTION
ns, errs = [20, 40, 80, 160, 320], []
for n in ns:
    g, mn, sc = fit_energy_gp(F_train[:n], E_train[:n])
    errs.append(np.sqrt(np.mean((g.predict((F_test - mn) / sc) - E_test)**2)) * 1000)
plt.loglog(ns, errs, "o-"); plt.xlabel("training points"); plt.ylabel("test RMS / meV"); plt.grid(True, which="both", alpha=0.3)
plt.title("learning curve"); plt.show()
print(dict(zip(ns, np.round(errs, 3))))
### END SOLUTION

# %% [markdown]
# ## 5. Do not trust it where there is no data
# **Exercise 6.** Stretch one bond from 1.0 to 2.0 Å (the other fixed at $r_e$, linear) and plot the GP energy with
# its ±2σ band. Compute a few DFT points along the same cut. Where does the GP fail, and does its uncertainty warn you?

# %%
r_cut = np.linspace(1.0, 2.0, 101)
F_cut = np.stack([r_cut, np.full_like(r_cut, X0[2, 2]), -np.ones_like(r_cut)], 1)
mu_cut, sd_cut = gp.predict((F_cut - f_mean) / f_scale, return_std=True)
r_chk = np.array([1.05, 1.25, 1.40, 1.55, 1.70, 1.85, 2.0])
### BEGIN SOLUTION
E_chk = [co2_energy_dipole(np.array([[0, 0, -r], [0, 0, 0], X0[2]]))[0] - E_min for r in r_chk]
### END SOLUTION
plt.fill_between(r_cut, mu_cut - 2 * sd_cut, mu_cut + 2 * sd_cut, alpha=0.3, label="GP ± 2σ")
plt.plot(r_cut, mu_cut, label="GP"); plt.plot(r_chk, E_chk, "ko", label="DFT")
plt.axvspan(F_train[:, :2].min(), F_train[:, :2].max(), color="0.9", zorder=-1, label="range of training data")
plt.xlabel("r₁ / Å"); plt.ylabel("E / eV"); plt.ylim(-0.5, 8); plt.legend(loc="upper left"); plt.show()

# %% [markdown]
# ## 6. Physical check: harmonic frequencies of the GP surface
# **Exercise 7.** Write `gp_energy(X)` for one geometry, then build the 9×9 Hessian of the GP surface by central
# finite differences (step 0.002 Å) and compute its frequencies as in Notebook 4. Compare with DFT.

# %%
def gp_energy(X):
    """GP energy (eV) at one or many geometries X (..., 3, 3)."""
    F = features(np.asarray(X)).reshape(-1, 3)
    return gp.predict((F - f_mean) / f_scale).reshape(np.shape(X)[:-2])

### BEGIN SOLUTION
step = 0.002
Hgp = np.zeros((9, 9))
x0 = X0.ravel()
for i in range(9):
    for j in range(i, 9):
        P = []
        for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            x = x0.copy(); x[i] += si * step; x[j] += sj * step; P.append(x.reshape(3, 3))
        e = gp_energy(np.array(P))
        Hgp[i, j] = Hgp[j, i] = (e[0] - e[1] - e[2] + e[3]) / (4 * step**2)
lam = np.linalg.eigvalsh(Hgp / np.sqrt(np.outer(m3, m3)))
wn_gp = lk.omega_to_wavenumber(np.sqrt(np.abs(lam[5:]) * lk.ACC))
### END SOLUTION
for a, b in zip(wn_gp, wn_dft):
    print(f"GP {a:7.1f}   DFT {b:7.1f}   difference {a - b:+5.1f} cm^-1")

# %% [markdown]
# ## 7. A model for the dipole moment
# The IR spectrum from MD needs the dipole at every step too. A simple, physical model: put a partial charge $q_1$ on
# O₁ and $q_2$ on O₂ (and $-(q_1+q_2)$ on C). Then
#
# $$\boldsymbol\mu = q_1\,\mathbf b_1 + q_2\,\mathbf b_2,\qquad \mathbf b_i = \mathbf x_{O_i} - \mathbf x_C.$$
#
# Let each charge depend linearly on the geometry, $q_1 = \mathbf c\cdot\boldsymbol\phi(r_1, r_2, \cos\theta)$ with
# $\boldsymbol\phi = (1,\ r_1 - r_e,\ r_2 - r_e,\ 1 + \cos\theta)$, and $q_2$ the same with $r_1 \leftrightarrow r_2$ (symmetry again).
# Then $\boldsymbol\mu$ is *linear* in the 4 coefficients $\mathbf c$, and they come from ordinary least squares.
#
# **Exercise 8.** Build the design matrix and fit $\mathbf c$ with `np.linalg.lstsq` on the training dipoles. Test it on
# your own dipoles. (1 e·Å = 4.803 D.)

# %%
r_e = X0[2, 2]

def phi(F):
    return np.stack([np.ones(len(F)), F[:, 0] - r_e, F[:, 1] - r_e, 1 + F[:, 2]], 1)

def dipole_design(X):
    """Design matrix A (3n x 4) with mu = A @ c, mu in debye."""
    F = features(X)
    b1, b2 = X[:, 0] - X[:, 1], X[:, 2] - X[:, 1]
    ### BEGIN SOLUTION
    P1, P2 = phi(F), phi(F[:, [1, 0, 2]])
    A = (P1[:, None, :] * b1[:, :, None] + P2[:, None, :] * b2[:, :, None]) * lk.DEBYE_PER_E_ANG
    ### END SOLUTION
    return A.reshape(-1, 4)

### BEGIN SOLUTION
c_dip, *_ = np.linalg.lstsq(dipole_design(X_train), mu_train.ravel(), rcond=None)
mu_pred = (dipole_design(X_test) @ c_dip).reshape(-1, 3)
rms_mu = np.sqrt(np.mean(np.sum((mu_pred - mu_test)**2, 1)))
### END SOLUTION
print("charge coefficients c =", np.round(c_dip, 4))
print(f"test RMS dipole error {rms_mu:.4f} D  (RMS dipole {np.sqrt(np.mean(np.sum(mu_test**2, 1))):.3f} D)")

# %% [markdown]
# **Question 8.** What is $q_1$ at the equilibrium geometry? Is the sign what you expect from electronegativity?
# How does the charge change when a bond stretches?
#
# *Your answer:*

# %% [markdown] instructor
# Reference fit: c = (−0.271, −0.155, +0.602, +0.034). At equilibrium q(O) = c₀ = −0.27 e on each O (C +0.54 e), as electronegativity suggests. c₁ < 0 and c₂ > 0: when bond 2 stretches and bond 1 shortens (the asymmetric stretch), O₂ becomes more negative and O₁ less negative, so charge flows toward the O atom that is moving away. This charge flux adds to the moving-charge term and is why ν₃ is so intense. Reference results: test RMS energy 0.017 meV (0.14 cm⁻¹, close to the DFT numerical noise); GP harmonic frequencies within 3 cm⁻¹ of DFT; dipole RMS 0.013 D with just 4 parameters. The learning curve drops from 0.09 meV (20 points) to a floor near 0.02 meV (the noise floor of the DFT grid). Along the stretch cut the GP follows DFT inside the data range and goes wrong outside it, with σ growing; a good place to discuss extrapolation. Students short of time can set N_TEST = 30.

# %% [markdown]
# ## 8. Save the surface for Notebook 6

# %%
with open("co2_ml_model.pkl", "wb") as f:
    pickle.dump({"gp": gp, "f_mean": f_mean, "f_scale": f_scale, "c_dip": c_dip, "r_e": r_e,
                 "X0": X0, "symbols": symbols, "masses": masses, "wn_dft": wn_dft}, f)
print("saved co2_ml_model.pkl")
