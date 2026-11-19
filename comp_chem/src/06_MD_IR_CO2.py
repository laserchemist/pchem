# %% [markdown]
# # Notebook 6 — Molecular dynamics on the learned surface, and the IR spectrum
#
# **Session 2, about 60 minutes**
#
# Now we let the nuclei move. **Classical molecular dynamics** solves Newton's equations
#
# $$m_i\,\ddot{\mathbf x}_i = \mathbf F_i = -\nabla_i E(\mathbf x)$$
#
# step by step, using the machine-learned surface for $E$. Nothing in MD is harmonic: the molecule stretches,
# bends, rotates, and trades energy between modes exactly as the surface dictates.
#
# You will:
# 1. write the **velocity Verlet** integrator and test it by **energy conservation**,
# 2. run a trajectory and **watch it in 3-D**,
# 3. compute the **IR spectrum** from the dipole moment along the trajectory, and compare with the harmonic
#    sticks of Notebook 4.
#
# ### Why the dipole gives the spectrum
# A molecule absorbs light at frequency ω when its dipole oscillates at ω. The absorption spectrum is the
# Fourier transform of the **dipole autocorrelation function** $\langle\boldsymbol\mu(0)\cdot\boldsymbol\mu(t)\rangle$.
# In practice (the Wiener–Khinchin theorem) this is $|\hat{\boldsymbol\mu}(\omega)|^2$, the squared Fourier transform of $\boldsymbol\mu(t)$ itself.
# For classical dynamics the absorption is $I(\omega) \propto \omega^2\,|\hat{\boldsymbol\mu}(\omega)|^2$.

# %%
import sys, os, time, pickle
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
import labkit as lk

def features(X):
    """(r1, r2, cos theta) for CO2 geometries X (..., 3, 3), atom order O, C, O (same as Notebook 5)."""
    b1 = X[..., 0, :] - X[..., 1, :]
    b2 = X[..., 2, :] - X[..., 1, :]
    r1 = np.linalg.norm(b1, axis=-1); r2 = np.linalg.norm(b2, axis=-1)
    return np.stack([r1, r2, np.sum(b1 * b2, axis=-1) / (r1 * r2)], axis=-1)

if os.path.exists("co2_ml_model.pkl"):
    model = pickle.load(open("co2_ml_model.pkl", "rb"))
    print("loaded your surface from Notebook 5")
else:   # fallback: refit the reference surface from the provided data (about 1 s)
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
    print("Notebook 5 output not found: refitted the reference surface")

gp, f_mean, f_scale, c_dip, r_e = model["gp"], model["f_mean"], model["f_scale"], model["c_dip"], model["r_e"]
X0, symbols, wn_harm = model["X0"], model["symbols"], model["wn_dft"]
masses = np.asarray(model["masses"])[:, None]          # shape (3, 1) so it broadcasts over x, y, z

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

print("E(X0) =", float(energy(X0)), "eV;  dipole(X0) =", dipole(X0), "D")

# %% [markdown]
# ## 1. Forces
# The force is minus the gradient of the energy. The simplest way to get it is a **central finite difference** in each
# of the 9 Cartesian coordinates:
#
# $$F_k = -\frac{E(\mathbf x + h\,\mathbf e_k) - E(\mathbf x - h\,\mathbf e_k)}{2h}.$$
#
# **Exercise 1.** Write `forces(X, h=1e-3)`. For speed, build all 18 displaced geometries as one array of shape
# `(18, 3, 3)` and call `energy` once. Check: at the minimum the forces are ≈ 0, and the total force on the molecule
# is zero at any geometry (Newton's third law, momentum conservation).

# %%
def forces(X, h=1e-3):
    """Forces (eV/A), shape (3, 3), on the atoms of geometry X."""
    ### BEGIN SOLUTION
    P = np.repeat(X[None], 18, axis=0)
    for k in range(9):
        P[2 * k].flat[k] += h
        P[2 * k + 1].flat[k] -= h
    e = energy(P)
    return -((e[0::2] - e[1::2]) / (2 * h)).reshape(3, 3)
    ### END SOLUTION

Xt = X0 + np.random.default_rng(0).normal(scale=0.03, size=(3, 3))
print("max |F| at the minimum:", np.abs(forces(X0)).max())
print("total force at a distorted geometry:", forces(Xt).sum(0))

# %% [markdown]
# ## 2. The velocity Verlet integrator
# With $\mathbf a = \mathbf F/m$ (times `lk.ACC` for units Å/fs²), one step of length $\Delta t$ is
#
# 1. $\mathbf v \leftarrow \mathbf v + \tfrac12 \Delta t\,\mathbf a(\mathbf x)$
# 2. $\mathbf x \leftarrow \mathbf x + \Delta t\,\mathbf v$
# 3. compute $\mathbf a(\mathbf x)$ at the new position
# 4. $\mathbf v \leftarrow \mathbf v + \tfrac12 \Delta t\,\mathbf a(\mathbf x)$
#
# It is time-reversible and conserves energy very well over long runs, which is why almost all MD codes use it.
#
# **Exercise 2.** Write `run_md(X, V, dt, n_steps, save_every)` returning the saved positions, the dipoles and the
# total energy at each saved step. Kinetic energy: $\tfrac12\sum m v^2$ in amu·Å²/fs², divided by `lk.ACC` for eV.

# %%
def kinetic(V):
    return 0.5 * np.sum(masses * V**2) / lk.ACC

def run_md(X, V, dt=0.5, n_steps=1000, save_every=1):
    """Velocity Verlet. Returns traj (n_saved, 3, 3), dipoles (n_saved, 3), total energies (n_saved,)."""
    X, V = X.copy(), V.copy()
    traj, mus, etot = [], [], []
    ### BEGIN SOLUTION
    F = forces(X)
    for step in range(n_steps):
        V += 0.5 * dt * F / masses * lk.ACC
        X += dt * V
        F = forces(X)
        V += 0.5 * dt * F / masses * lk.ACC
        if step % save_every == 0:
            traj.append(X.copy()); mus.append(dipole(X)); etot.append(energy(X) + kinetic(V))
    ### END SOLUTION
    return np.array(traj), np.array(mus), np.array(etot)

# %% [markdown]
# ## 3. Starting conditions at a temperature T
# For a thermal start we give each normal mode a random harmonic displacement with $\langle\tfrac12\omega^2Q^2\rangle = \tfrac12 k_BT$,
# and each atom random **Maxwell–Boltzmann** velocities, $v \sim \mathcal N(0, \sqrt{k_BT/m})$ per direction. Then we remove
# the centre-of-mass velocity so the molecule does not drift away. This function is provided.

# %%
h = np.load("co2_harmonic.npz" if os.path.exists("co2_harmonic.npz") else lk.data_path("co2_harmonic.npz"))
modes, omega = h["modes"], 2 * np.pi * lk.C_CM_PER_FS * h["wn"]
m3 = np.repeat(masses.ravel(), 3)

def thermal_start(T, rng):
    kT = lk.KB_EV * T
    Q = rng.normal(size=4) * np.sqrt(kT * lk.ACC) / omega
    X = X0 + ((modes @ Q) / np.sqrt(m3)).reshape(3, 3)
    V = rng.normal(size=(3, 3)) * np.sqrt(kT * lk.ACC / masses)
    V -= (masses * V).sum(0) / masses.sum()
    return X, V

# %% [markdown]
# **Exercise 3. Choosing the time step.** The fastest vibration (ν₃ ≈ 2470 cm⁻¹) has a period of 13.5 fs.
# Run 1 ps from the same 300 K start with Δt = 0.25, 0.5, 1, 2 and 4 fs. Plot $E_\text{tot}(t) - E_\text{tot}(0)$ in meV, and
# on log–log axes the size of the energy fluctuation (`np.ptp`) against Δt. Which time steps are acceptable? What is the slope?

# %%
rng = np.random.default_rng(7)
Xs, Vs = thermal_start(300, rng)
t0 = time.time()
dts, spread = (4.0, 2.0, 1.0, 0.5, 0.25), []
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
### BEGIN SOLUTION
for dt in dts:                      # largest first, so the small-step curves are drawn on top
    every = max(1, int(1 / dt))
    _, _, et = run_md(Xs, Vs, dt, int(1000 / dt), save_every=every)
    ax[0].plot(np.arange(len(et)) * every * dt, (et - et[0]) * 1000, label=f"Δt = {dt} fs")
    spread.append(np.ptp(et) * 1000)
ax[0].set_ylim(-6, 6); ax[0].set_xlabel("t / fs"); ax[0].set_ylabel("E_tot − E_tot(0) / meV"); ax[0].legend(fontsize=8)
ax[1].loglog(dts, spread, "o-"); ax[1].loglog(dts, spread[-1] * (np.array(dts) / dts[-1])**2, "k--", label="∝ Δt²")
ax[1].set_xticks(dts, [str(d) for d in dts]); ax[1].minorticks_off()
ax[1].set_xlabel("Δt / fs"); ax[1].set_ylabel("energy fluctuation / meV"); ax[1].legend()
plt.tight_layout(); plt.show()
### END SOLUTION
print(f"({time.time() - t0:.0f} s)")

# %% [markdown]
# **Question 3.** Why does the energy blow up when Δt is a sizeable fraction of the vibrational period? What is the cost
# of a smaller time step?
#
# *Your answer:*

# %% [markdown] instructor
# Verlet is accurate when the step resolves the fastest motion: rule of thumb Δt ≲ period/10–20. The energy fluctuation grows as Δt² (Verlet is a second-order method): about 0.1 meV at 0.25 fs, about 1 meV at 1 fs, several meV at 2 fs, and at 4 fs (≈ period/3) it is unstable or wildly inaccurate. The floor at the smallest steps comes from finite-difference forces on a GP whose predictions carry ~µeV round-off; an analytic GP gradient would remove it (extension for strong students). Cost scales as 1/Δt. Note: a bounded fluctuation in E_tot is normal for Verlet (it conserves a nearby "shadow" Hamiltonian); a steady drift is the warning sign.

# %% [markdown]
# ## 4. A trajectory, and the movie
# **Exercise 4.** Run 10 ps at 300 K with Δt = 0.5 fs, saving every step. Plot $r_1(t)$, $r_2(t)$ and the O–C–O angle for the
# first 500 fs. Then watch the first 2 ps.

# %%
rng = np.random.default_rng(11)
X_init, V_init = thermal_start(300, rng)
t0 = time.time()
traj, mu, et = run_md(X_init, V_init, dt=0.5, n_steps=20000, save_every=1)
print(f"10 ps in {time.time() - t0:.0f} s; energy range {1000 * np.ptp(et):.2f} meV")
times = np.arange(len(traj)) * 0.5
F = features(traj)
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(times[:1000], F[:1000, 0], label="r₁"); ax[0].plot(times[:1000], F[:1000, 1], label="r₂")
ax[0].set_ylabel("C–O / Å"); ax[0].legend()
ax[1].plot(times[:1000], np.degrees(np.arccos(np.clip(F[:1000, 2], -1, 1)))); ax[1].set_ylabel("O–C–O / deg")
ax[1].set_xlabel("t / fs"); plt.show()

# %%
# the movie: every 2nd step (1 fs per frame) for the first 2 ps; drag to rotate, scroll to zoom
lk.animate_trajectory(symbols, traj[:4000], stride=2, interval=30)

# %% [markdown]
# **Question 4.** In the plots, which motions can you identify, and with roughly what periods? (Compare with the harmonic
# wavenumbers: period = 1/(c ν̃) = 33 356 fs / ν̃[cm⁻¹].) In the movie, what does the molecule do besides vibrate?
#
# *Your answer:*

# %% [markdown]
# ## 5. The dipole and the IR spectrum
# **Exercise 5.** Plot the three dipole components for the first 500 fs. Which motions make them oscillate?

# %%
### BEGIN SOLUTION
for k, lab in enumerate("xyz"):
    plt.plot(times[:1000], mu[:1000, k], label=f"μ_{lab}")
plt.xlabel("t / fs"); plt.ylabel("μ / D"); plt.legend(); plt.show()
### END SOLUTION

# %% [markdown]
# **Exercise 6.** Write `ir_spectrum(mu, dt_fs)`:
# 1. subtract the mean dipole,
# 2. multiply by a Hann window (`np.hanning`) so the sharp ends of the record do not add ripples,
# 3. FFT each component with `np.fft.rfft` (pad to 4× the length with `n=` for a smoother curve) and sum $|\hat\mu|^2$ over x, y, z,
# 4. convert the frequency axis (`np.fft.rfftfreq`, in fs⁻¹) to cm⁻¹ by dividing by c (`lk.C_CM_PER_FS`),
# 5. multiply by $\tilde\nu^2$.
#
# Plot the spectrum from 0 to 3000 cm⁻¹ with the harmonic sticks from Notebook 4.

# %%
def ir_spectrum(mu, dt_fs):
    """Classical IR absorption spectrum from a dipole time series. Returns (wavenumber cm^-1, intensity, arbitrary units)."""
    ### BEGIN SOLUTION
    m = (mu - mu.mean(0)) * np.hanning(len(mu))[:, None]
    n = 4 * len(m)
    S = np.sum(np.abs(np.fft.rfft(m, n=n, axis=0))**2, axis=1)
    nu = np.fft.rfftfreq(n, d=dt_fs) / lk.C_CM_PER_FS
    return nu, S * nu**2
    ### END SOLUTION

nu, I = ir_spectrum(mu, 0.5)
sel = nu < 3000
plt.figure(figsize=(8, 3.5))
plt.plot(nu[sel], I[sel] / I[sel].max(), label="MD, 300 K, one 10 ps trajectory")
plt.vlines(wn_harm, 0, 1, color="r", ls=":", label="harmonic (Notebook 4)")
plt.xlabel("wavenumber / cm⁻¹"); plt.ylabel("absorbance (normalised)"); plt.legend(); plt.show()
for lo, hi in ((500, 800), (1200, 1500), (2200, 2700)):
    w = (nu > lo) & (nu < hi)
    print(f"{lo}-{hi} cm^-1: peak at {nu[w][np.argmax(I[w])]:.0f} cm^-1, relative intensity {I[w].max() / I[sel].max():.1e}")

# %% [markdown]
# **Question 6.** Which normal modes appear in the MD spectrum and which does not? Is this the same conclusion as the
# harmonic analysis, even though nothing in MD is harmonic? Why do the MD peaks sit slightly away from the harmonic sticks?
#
# *Your answer:*

# %% [markdown] instructor
# Reference run (one 10 ps trajectory): bend peak 653 cm⁻¹ and asymmetric stretch 2481 cm⁻¹ appear; the symmetric stretch (≈ 1389) is absent (relative intensity ~10⁻⁸ or less, numerical noise), because the selection rule is a symmetry statement that holds for any amplitude, not a harmonic one. Small shifts from the sticks: anharmonicity (the GP surface is the full DFT surface), sampled at 300 K (classical; no zero-point energy, so classical MD at low T gives nearly harmonic frequencies, not fundamentals), plus the finite resolution of 10 ps (≈ 3 cm⁻¹). A single trajectory has a fixed rotational angular momentum, so each band may show a split doublet — see next section.

# %% [markdown]
# ## 6. Rotation shapes the band: averaging over trajectories
# A single molecule rotates at one rate, which splits each band into two peaks (rotation adds and subtracts from the
# vibrational frequency). A real gas has a thermal distribution of rotational speeds.
#
# **Exercise 7.** Average the spectrum of 24 independent 300 K trajectories (4 ps each, about 2 minutes) and zoom in on the
# asymmetric stretch (2400–2550 cm⁻¹). You should see the **P and R branches** of the gas-phase spectrum. For a linear molecule
# their maxima are separated by about $2.36\sqrt{B\,T}$ cm⁻¹ with $B$ = 0.39 cm⁻¹ for CO₂. Compare.

# %%
def averaged_spectrum(T, n_traj=24, t_ps=4, dt=0.5, seed=0):
    rng = np.random.default_rng(seed)
    acc = 0
    for k in range(n_traj):
        X, V = thermal_start(T, rng)
        ### BEGIN SOLUTION
        _, m, _ = run_md(X, V, dt, int(t_ps * 1000 / dt), save_every=2)
        nu, I = ir_spectrum(m, 2 * dt)
        acc = acc + I
        ### END SOLUTION
    return nu, acc / n_traj

t0 = time.time()
nu300, I300 = averaged_spectrum(300)
print(f"({time.time() - t0:.0f} s)")
w = (nu300 > 2400) & (nu300 < 2550)
plt.plot(nu300[w], I300[w] / I300[w].max()); plt.axvline(wn_harm[3], color="r", ls=":", label="harmonic ν₃")
plt.xlabel("wavenumber / cm⁻¹"); plt.title("ν₃ band, 300 K, 24 trajectories"); plt.legend(); plt.show()
print(f"expected P-R separation: {2.36 * np.sqrt(0.39 * 300):.0f} cm^-1")

# %% [markdown]
# ## 7. A hot molecule
# **Exercise 8.** Repeat at 1500 K. How do the bands change (position, width, new features)? Make the movie for the hot
# molecule too.

# %%
t0 = time.time()
### BEGIN SOLUTION
nu1500, I1500 = averaged_spectrum(1500)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for nu_, I_, lab in ((nu300, I300, "300 K"), (nu1500, I1500, "1500 K")):
    for a, (lo, hi) in zip(ax, ((550, 760), (2300, 2560))):
        w = (nu_ > lo) & (nu_ < hi)
        a.plot(nu_[w], I_[w] / I_[w].max(), label=lab)
for a, k in zip(ax, (0, 3)):
    a.axvline(wn_harm[k], color="r", ls=":"); a.set_xlabel("wavenumber / cm⁻¹"); a.legend()
ax[0].set_title("bend ν₂"); ax[1].set_title("asymmetric stretch ν₃"); plt.show()
### END SOLUTION
print(f"({time.time() - t0:.0f} s)")

# %%
X, V = thermal_start(1500, np.random.default_rng(3))
hot, _, _ = run_md(X, V, 0.5, 4000, save_every=1)
lk.animate_trajectory(symbols, hot, stride=2, interval=30)

# %% [markdown] instructor
# Reference (24 × 4 ps): at 300 K the ν₃ band has branches near 2462 and 2479 cm⁻¹, centred about 5–10 cm⁻¹ below the harmonic 2472 (classical anharmonic shift, with the bend thermally excited); the two-branch envelope is the classical P/R contour (expected separation ≈ 26 cm⁻¹; finite sampling makes it ~20). With only 24 trajectories the contour is still noisy, which is worth pointing out. Expected: at 1500 K both bands broaden and shift to lower wavenumber (ν₃ spreads over 2320–2450 cm⁻¹) (the surface softens at larger amplitude; anharmonic coupling between bend and stretches); the P/R branches spread (higher J). The bend region may show extra structure from the Fermi coupling between ν₁ and 2ν₂ (1389 ≈ 2 × 654), which classical mechanics also exhibits. If time allows, connect to the CH₂OO research example in which the same MD/IR machinery was used for a molecule with 9 modes.

# %% [markdown]
# ## 8. Save a figure and the trajectory
# A static film strip is useful for your report. The `.xyz` file can be opened in other viewers (VMD, Avogadro, Jmol).

# %%
fig, ax = plt.subplots(figsize=(9, 1.8))
lk.trajectory_strip(symbols, traj, times, frames=[0, 10, 20, 30, 40, 50], plane=(2, 0), ax=ax)
plt.show()
with open("co2_300K.xyz", "w") as f:
    for i in range(0, 4000, 2):
        f.write(f"3\nt = {times[i]:.1f} fs\n")
        for s, x in zip(symbols, traj[i]):
            f.write(f"{s} {x[0]:.5f} {x[1]:.5f} {x[2]:.5f}\n")
print("wrote co2_300K.xyz")
