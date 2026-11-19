# %% [markdown]
# # Notebook 4 — CO₂: normal modes and why it absorbs infrared light
#
# **Session 2, about 60 minutes**
#
# CO₂ is only 0.04 % of the atmosphere, yet it is the main long-lived greenhouse gas. It has no permanent dipole
# moment, and it is linear and symmetric like O₂. Why does it absorb infrared light when O₂ does not?
#
# A molecule with $N$ atoms has $3N$ coordinates. For a **linear** molecule, 3 are translations and 2 are
# rotations, leaving $3N-5 = 4$ **vibrations** for CO₂. In the **harmonic approximation** the energy near
# the minimum is
#
# $$E \approx E_0 + \tfrac12 \sum_{ij} H_{ij}\, \Delta x_i\, \Delta x_j, \qquad H_{ij} = \frac{\partial^2 E}{\partial x_i\,\partial x_j}.$$
#
# $H$ is the **Hessian** (force-constant matrix). The vibrations that move independently are the
# **normal modes**: the eigenvectors of the *mass-weighted* Hessian $H_{ij}/\sqrt{m_i m_j}$, whose eigenvalues are $\omega^2$.
#
# In this notebook you will:
# 1. find the equilibrium geometry,
# 2. build and diagonalise the mass-weighted Hessian,
# 3. watch each normal mode in 3-D,
# 4. compute each mode's IR intensity from the change of the dipole, $|\partial\boldsymbol\mu/\partial Q|^2$.

# %%
import sys, time
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pyscf import gto, dft
import labkit as lk

symbols = ["O", "C", "O"]
masses = np.array([lk.MASS[s] for s in symbols])

def co2_mf(X, basis="def2-svp", xc="b3lyp"):
    """Run B3LYP on CO2 at Cartesian geometry X (3x3, angstrom); return the converged PySCF object."""
    mol = gto.M(atom=[[s, tuple(x)] for s, x in zip(symbols, X)], basis=basis, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc
    mf.kernel()
    return mf

def linear_co2(r):
    return np.array([[0, 0, -r], [0, 0, 0], [0, 0, r]], float)

# %% [markdown]
# ## 1. Equilibrium geometry
# By symmetry the minimum is linear with two equal C=O bonds, so we only need to minimise over one bond length.
#
# **Exercise 1.** Use `scipy.optimize.minimize_scalar` (method `"bounded"`, bounds 1.10–1.22 Å) to find $r_e$.
# Experiment: 1.1600 Å.

# %%
t = time.time()
### BEGIN SOLUTION
res = minimize_scalar(lambda r: co2_mf(linear_co2(r)).e_tot, bounds=(1.10, 1.22), method="bounded",
                      options={"xatol": 1e-5})
r_e = res.x
### END SOLUTION
X0 = linear_co2(r_e)
print(f"r_e(C=O) = {r_e:.4f} A   ({time.time() - t:.0f} s)")
lk.show_molecule(symbols, X0, labels=True)

# %% [markdown]
# ## 2. The Hessian
# PySCF computes the Hessian analytically: `mf.Hessian().kernel()` returns an array of shape
# `(N, N, 3, 3)` in hartree/bohr², where element `[a, b, i, j]` is $\partial^2E/\partial x_{a,i}\,\partial x_{b,j}$
# for atoms a, b and Cartesian directions i, j.
#
# **Exercise 2.**
# (a) Reshape it into a $9\times 9$ matrix with rows ordered (atom, direction), and convert it to eV/Å².
# (b) **Check one element yourself**: displace the carbon atom along z by ±0.005 Å, and compare the
# finite-difference second derivative $[E(+h) - 2E(0) + E(-h)]/h^2$ with $H_{zz}$ for carbon.

# %%
t = time.time()
mf0 = co2_mf(X0)
H4 = mf0.Hessian().kernel()
print(f"analytic Hessian: shape {H4.shape} ({time.time() - t:.0f} s)")
### BEGIN SOLUTION
H = H4.transpose(0, 2, 1, 3).reshape(9, 9) * lk.HARTREE_TO_EV / lk.BOHR_TO_ANG**2
### END SOLUTION
print("symmetric:", np.allclose(H, H.T, atol=1e-6))

### BEGIN SOLUTION
h = 0.005
Ep = co2_mf(X0 + np.array([[0, 0, 0], [0, 0, h], [0, 0, 0]])).e_tot
Em = co2_mf(X0 - np.array([[0, 0, 0], [0, 0, h], [0, 0, 0]])).e_tot
E0 = mf0.e_tot
fd = (Ep - 2 * E0 + Em) / h**2 * lk.HARTREE_TO_EV
### END SOLUTION
print(f"H[C_z, C_z]: analytic {H[5, 5]:.2f} eV/A^2, finite difference {fd:.2f} eV/A^2")

# %% [markdown]
# ## 3. Normal modes and frequencies
# **Exercise 3.** Mass-weight the Hessian, $\tilde H_{ij} = H_{ij}/\sqrt{m_i m_j}$, diagonalise it with `np.linalg.eigh`,
# and convert the eigenvalues $\lambda = \omega^2$ to wavenumbers. (Units: $\omega$ in rad/fs is
# $\sqrt{\lambda \times \texttt{ACC}}$; use `lk.omega_to_wavenumber`.) Report negative eigenvalues as negative wavenumbers.
# How many are close to zero, and why?

# %%
m3 = np.repeat(masses, 3)                         # mass of each of the 9 coordinates
### BEGIN SOLUTION
Hmw = H / np.sqrt(np.outer(m3, m3))
lam, L = np.linalg.eigh(Hmw)
wn_all = lk.omega_to_wavenumber(np.sign(lam) * np.sqrt(np.abs(lam) * lk.ACC))
### END SOLUTION
print(np.round(wn_all, 1))

# %% [markdown]
# The 5 smallest are the translations and rotations. They should be exactly zero, but the DFT
# integration grid introduces errors of ±20 cm⁻¹. The 4 vibrations are the last four.

# %%
wn = wn_all[5:]
modes = L[:, 5:]                                   # mass-weighted eigenvectors, columns
disp = modes / np.sqrt(m3)[:, None]                # Cartesian displacement pattern of each mode
for k in range(4):
    print(f"mode {k}: {wn[k]:7.1f} cm^-1   displacement (x,y,z per atom):")
    print(np.round(disp[:, k].reshape(3, 3), 3))

# %% [markdown]
# **Exercise 4.** Watch each mode. Name them: symmetric stretch, asymmetric stretch, and the two
# degenerate bends. Why are there two bends but only one of each stretch?

# %%
for k in range(4):
    print(f"mode {k}: {wn[k]:.0f} cm^-1")
    lk.animate_mode(symbols, X0, disp[:, k], amplitude=0.25)

# %% [markdown]
# ## 4. Infrared intensities
# The IR intensity of mode $k$ is proportional to $|\partial\boldsymbol\mu/\partial Q_k|^2$, where $Q_k$ is the
# mass-weighted normal coordinate. Move the molecule a small step along the mode,
# $\mathbf X_\pm = \mathbf X_0 \pm \delta\,\mathbf L_k/\sqrt{m}$, and take a central difference:
#
# $$\frac{\partial\boldsymbol\mu}{\partial Q_k} \approx \frac{\boldsymbol\mu(\mathbf X_+) - \boldsymbol\mu(\mathbf X_-)}{2\delta}.$$
#
# With $\mu$ in D, $Q$ in Å·amu$^{1/2}$, the intensity in the usual units is
# $A_k\,[\text{km/mol}] = 42.256\,|\partial\boldsymbol\mu/\partial Q_k|^2$.
#
# **Exercise 5.** Write `ir_intensity(k, delta=0.01)` and compute $A_k$ for the four modes.

# %%
def dipole(X):
    return co2_mf(X).dip_moment(unit="Debye", verbose=0)

def ir_intensity(k, delta=0.01):
    """IR intensity (km/mol) of normal mode k by central finite difference of the dipole."""
    ### BEGIN SOLUTION
    dX = (delta * disp[:, k]).reshape(3, 3)
    dmu = (dipole(X0 + dX) - dipole(X0 - dX)) / (2 * delta)
    return 42.256 * np.sum(dmu**2)
    ### END SOLUTION

t = time.time()
A = np.array([ir_intensity(k) for k in range(4)])
print(f"({time.time() - t:.0f} s)")
for k in range(4):
    print(f"mode {k}: {wn[k]:7.1f} cm^-1   A = {A[k]:7.1f} km/mol")

# %% [markdown]
# Experiment (gas phase): bend ν₂ = 667 cm⁻¹ (2 × 24 km/mol), symmetric stretch ν₁ ≈ 1333 cm⁻¹ (inactive),
# asymmetric stretch ν₃ = 2349 cm⁻¹ (548 km/mol).
#
# **Question 5.** Explain from symmetry why the symmetric stretch has zero intensity while the bend and the
# asymmetric stretch are strong. (Sketch the dipole at the two turning points of each vibration.)
#
# *Your answer:*

# %% [markdown] instructor
# In the symmetric stretch the inversion centre is kept at every instant, so μ = 0 throughout (as in O₂): IR-inactive, Raman-active. The asymmetric stretch and the bend break the inversion symmetry; the two C=O bond dipoles no longer cancel, and μ oscillates along z (ν₃) or perpendicular to the axis (ν₂). Reference run (B3LYP/def2-SVP): r_e = 1.163 Å; 654 (bend, ×2), 1389, 2472 cm⁻¹; A = 2 × 33, 0, 614 km/mol. The analytic H[C_z,C_z] = 196.18 eV/Å² matches the finite difference to 4 figures. Harmonic B3LYP/def2-SVP overestimates ν₃ by ~5 % (part harmonic, part method/basis); ν₁ is not directly observed because of the Fermi resonance with 2ν₂ (1285/1388 cm⁻¹ Raman doublet), which is a nice extension topic.

# %% [markdown]
# ## 5. The greenhouse connection
# The Earth's surface (≈ 288 K) radiates like a black body. In wavenumber units the Planck distribution is
#
# $$B(\tilde\nu) \propto \frac{\tilde\nu^3}{e^{hc\tilde\nu/k_BT} - 1}.$$
#
# **Exercise 6.** Plot your stick spectrum together with the Planck curve at 288 K (scaled to fit).
# Which CO₂ band sits under the Earth's emission?

# %%
### BEGIN SOLUTION
nu = np.linspace(100, 3000, 600)
x = nu / lk.EV_TO_CM1 / (lk.KB_EV * 288)
B = nu**3 / np.expm1(x)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.fill_between(nu, B / B.max() * A.max(), color="#f2c49b", alpha=0.6, label="Earth emission, 288 K (scaled)")
ax.vlines(wn, 0, A, color="#1f4e79", lw=3, label="CO₂ harmonic IR, B3LYP")
ax.set_xlabel("wavenumber / cm⁻¹"); ax.set_ylabel("intensity / km mol⁻¹"); ax.legend()
plt.show()
### END SOLUTION

# %% [markdown]
# **Question 6.** The bend at 667 cm⁻¹ (15 µm) sits near the peak of the Earth's thermal emission. Explain in two or
# three sentences why this one vibration matters so much for climate, and why ν₃, though 10× stronger, matters less.
#
# *Your answer:*

# %% [markdown] instructor
# The Planck curve at 288 K peaks near 600 cm⁻¹ (≈ 17 µm in wavenumber representation), so the bend absorbs where the surface emits most strongly; the band is already saturated at its centre, so added CO₂ widens the band (the logarithmic dependence of forcing on concentration). ν₃ at 2349 cm⁻¹ lies in the far tail of the 288 K emission, where very little energy is emitted.

# %% [markdown]
# ## 6. Save the results for Notebooks 5 and 6

# %%
np.savez("co2_harmonic.npz", symbols=np.array(symbols), X0=X0, wn=wn, disp=disp, modes=modes,
         intensity=A, masses=masses)
print("saved co2_harmonic.npz")
