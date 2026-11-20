# LAB
# %% [markdown]
# # Computational Lab 3: Quantum Vibrational Levels of CO on a Computed Curve
# *CHEM 3376 — Physical Chemistry Laboratory: Data Analysis & Computation*
#
# The harmonic wavenumber from a curvature (Lab 2) is not what a spectrometer measures. Real bonds are **anharmonic**: they
# get softer as they stretch and eventually break. The CO fundamental is observed at **2143 cm⁻¹**, about 27 cm⁻¹ below the
# harmonic value $\omega_e$ = 2170 cm⁻¹.
#
# In this lab you solve the vibrational Schrödinger equation **exactly on a grid** for the CCSD(T) curve from Lab 2:
#
# $$\left[-\frac{\hbar^2}{2\mu}\frac{d^2}{dr^2} + V(r)\right]\psi_v(r) = E_v\,\psi_v(r)$$
#
# and then use the wavefunctions to predict:
#
# 1. the **energy levels**, the zero-point energy and the fundamental;
# 2. the **wavefunctions**, and how they turn classical at high $v$;
# 3. the **anharmonicity constant** $\omega_e x_e$;
# 4. the **intensities** of the fundamental and the first overtone;
# 5. the **isotope shift** of ¹³C¹⁶O.
#
# Every calculation here takes seconds. If you didn't finish Lab 2, a reference copy of its curve is used automatically.
#
# <br>**<center>Learning Goals**
#
# |Area|Concept|
# |---|---|
# |Quantum mechanics|The Schrödinger equation as a matrix eigenvalue problem; zero-point energy; tunnelling; the correspondence principle|
# |Spectroscopy|Anharmonicity, the fundamental and overtones; the Birge–Sponer form $G(v)$; transition dipoles and selection rules|
# |Isotopes|The Born–Oppenheimer approximation: same potential, different mass|
# |Numerical methods|Finite differences; testing a method on a problem with a known answer|
# |Python|`np.diag` and `np.eye` to build matrices; `np.linalg.eigh`; splines; numerical integrals as sums|

# %% [markdown]
# ---
# ## 📄 Abstract
#
# <div style="border:2px solid #1f4e79; border-radius:8px; padding:10px 14px; background:#f3f7fb; color:#1a1a1a;">
#
# **Write this section last**, after you have finished the lab. Your abstract goes in the cell below, replacing `ANSWER`.
#
# In **150–250 words**, as in a journal article: state the purpose; name the method (finite-difference solution on the
# CCSD(T)/cc-pVTZ curve, B3LYP dipole curve); give your **key numbers** with experiment beside them (fundamental, zero-point
# energy, ω_e and ω_e x_e, the overtone/fundamental intensity ratio, the ¹³C isotope shift); and finish with the main conclusion.
# </div>

# %% [markdown] answer
# The vibrational Schrödinger equation of carbon monoxide was solved by the finite-difference method on the CCSD(T)/cc-pVTZ potential-energy curve, to compare the anharmonic levels with the measured infrared spectrum. The method was first verified on a harmonic oscillator, where it reproduces E_v/ħω = v + ½ to better than 0.1 %. For ¹²C¹⁶O the zero-point energy is 1082 cm⁻¹ (experiment 1081.6 cm⁻¹) and the fundamental is 2145.4 cm⁻¹ (experiment 2143.3 cm⁻¹); a fit of the lowest five levels gives ω_e = 2171.4 cm⁻¹ and ω_e x_e = 13.0 cm⁻¹ (experiment 2169.8 and 13.29 cm⁻¹). The wavefunctions lean to longer bond lengths at higher v, because the outer wall of the potential is softer, and they extend past the classical turning points. With the B3LYP dipole curve the 0→2 overtone is about 210 times weaker than the fundamental, the right order of magnitude (observed about 1/150), because overtones are allowed only by the anharmonicity of the potential and the curvature of the dipole. For ¹³C¹⁶O the fundamental shifts by −47.3 cm⁻¹ (experiment −47.2 cm⁻¹), as the Born–Oppenheimer approximation predicts for an unchanged potential and a heavier reduced mass. An accurate potential plus an exact one-dimensional solution reproduces the CO spectrum to within a few cm⁻¹.

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
ver = '2026Comp03V1'

# %% [markdown]
# ### 🧰 Standard imports
# `numpy`, `matplotlib`, a cubic spline from SciPy, and `labkit`, this course's helper package (constants and masses).
# The last lines load this lab's **answer checks**: after each question, run `lab.check(...)`. A ✅ means you got it;
# a ❌ tells you what to fix. Fix the cell and run the check again, as often as you like.

# %%
# Standard imports for this lab
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
for _up in ("..", os.path.join("..", "..")):          # labkit is in comp_chem/, the folder above this lab
    if os.path.isdir(os.path.join(_up, "labkit")):
        sys.path.insert(0, os.path.abspath(_up))
        break
import labkit as lk

# Answer checks for this lab: the checker (pchem_check.py) and the answer key (answers_comp03.json) are in this folder
from pchem_check import Checker
lab = Checker('comp03', 'answers_comp03.json')

# %% [markdown]
# Now run the cell below to enter your name.

# %%
# Enter your name as a string, e.g. name = 'Linus Pauling'
name = 'Instructor'  # SOLUTION

# %% [markdown]
# **Just run the cell below.** It loads the CCSD(T)/cc-pVTZ curve and the B3LYP dipole curve that you saved in Lab 2 (or
# the reference copy), and sets up the masses. Energies are in eV relative to the minimum; lengths in Å.

# %%
# Just run this cell: the curve from Lab 2, constants and masses
d = np.load(lk.find_result("co_ccsdt_curve.npz"))
r_pts, V_pts, mu_pts = d["r"], d["E_eV"], d["mu_D"]
print(len(r_pts), "points from", r_pts[0], "to", r_pts[-1], "Å")

EV_TO_CM1 = lk.EV_TO_CM1          # cm⁻¹ per eV (8065.54)
HBAR = lk.HBAR_EV_FS              # ħ in eV·fs
m_C, m_O = lk.MASS["C"], lk.MASS["O"]
mu_CO = m_C * m_O / (m_C + m_O)   # reduced mass, amu

# %% [markdown]
# ## 1. The Schrödinger Equation as a Matrix

# %% [markdown]
# ### 1.1 A smooth potential
# The grid needs $V$ at many more points than were computed. A **cubic spline** passes smoothly through the computed points;
# `V(r)` then works for any `r` (or an array of them). **Just run the cell.**

# %%
# Just run this cell: splines through the computed points
V = CubicSpline(r_pts, V_pts)
mu_of_r = CubicSpline(r_pts, mu_pts)
r_plot = np.linspace(r_pts[0], r_pts[-1], 300)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(r_pts, V_pts, "o", label="CCSD(T)/cc-pVTZ"); ax[0].plot(r_plot, V(r_plot), label="spline")
ax[0].set_ylabel("V / eV"); ax[0].legend()
ax[1].plot(r_pts, mu_pts, "o", label="B3LYP"); ax[1].plot(r_plot, mu_of_r(r_plot), label="spline")
ax[1].set_ylabel("μ_z / D"); ax[1].legend()
for a in ax:
    a.set_xlabel("r(C–O) / Å")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 1.2 The finite-difference method
# Put $r$ on an evenly spaced grid $r_i$ with spacing $h$. The second derivative becomes
#
# $$\psi''(r_i) \approx \frac{\psi_{i+1} - 2\psi_i + \psi_{i-1}}{h^2},$$
#
# so the Hamiltonian becomes a **matrix**: $V(r_i) + 2t$ on the diagonal and $-t$ on the two neighbouring diagonals, where
#
# $$ t = \frac{\hbar^2}{2\mu h^2} . $$
#
# Its **eigenvalues** are the energy levels and its **eigenvectors** are the wavefunctions on the grid
# ([Atkins, Topic 7E (the harmonic oscillator) and 11C (anharmonicity)](https://global.oup.com/academic/product/atkins-physical-chemistry-9780198769866)).
#
# **Units.** With $\hbar$ in eV·fs, $\mu$ in amu and $h$ in Å, $\hbar^2/(\mu h^2)$ needs the same conversion as Newton's law
# (1 amu·Å²/fs² = 1/`lk.ACC` eV):
#
# $$ t\ [\text{eV}] = \frac{\hbar^2\,[\text{eV}^2\text{fs}^2]\times \texttt{lk.ACC}}{2\,\mu\,[\text{amu}]\,h^2\,[\text{Å}^2]}. $$

# %% [markdown]
# ### <font color=blue> **Question 1.1.** </font>
# Complete `vib_levels`: write the hopping energy `t` and the Hamiltonian matrix `H`. The cell then **tests the method on a
# harmonic oscillator**, whose levels are known exactly: $E_v = \hbar\omega(v + \tfrac12)$.
#
# 🐍 **Python tip — building matrices.** `np.diag(a)` makes a square matrix with the array `a` on its diagonal.
# `np.eye(n, k=1)` is an n × n matrix of zeros with ones on the diagonal just *above* the main one (`k=-1`: just below).
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `t = HBAR**2 * lk.ACC / (2 * mu * h**2)` and `H = np.diag(V(r) + 2 * t) - t * np.eye(n, k=1) - t * np.eye(n, k=-1)`.
# </details>

# %%
def vib_levels(V, mu, r_min=0.90, r_max=1.60, n=600):
    """Finite-difference vibrational levels for potential V(r) (eV) and reduced mass mu (amu).
    Returns the grid r (Å), the energies E (eV) and the wavefunctions psi (one per column)."""
    r = np.linspace(r_min, r_max, n)
    h = r[1] - r[0]
    t = HBAR**2 * lk.ACC / (2 * mu * h**2)  # SOLUTION
    H = np.diag(V(r) + 2 * t) - t * np.eye(n, k=1) - t * np.eye(n, k=-1)  # SOLUTION
    E, psi = np.linalg.eigh(H)
    psi = psi / np.sqrt(h)                     # normalise: sum of |psi|² h = 1
    return r, E, psi

# test on a harmonic oscillator, k = 120 eV/Å², centred at 1.13 Å
k_test = 120.0
_, E_harm, _ = vib_levels(lambda x: 0.5 * k_test * (x - 1.13)**2, mu_CO)
hw = HBAR * np.sqrt(k_test / mu_CO * lk.ACC)
print("E_v / ħω for v = 0–3:", np.round(E_harm[:4] / hw, 4), "  (exact: 0.5, 1.5, 2.5, 3.5)")

# %%
lab.check('1.1')

# %% [markdown]
# ### <font color=blue> **Question 1.2.** </font>
# Now the real CO curve. Compute
# - `wn`, the energies of all the levels **above the ground state**, $E_v - E_0$, in cm⁻¹ (so `wn[1]` is the fundamental);
# - `zpe`, the **zero-point energy** $E_0 - V_\text{min}$ in cm⁻¹. `V_min` is the bottom of the spline curve, found on the fine grid
#   (the lowest *computed* point, `V_pts.min()`, is up to 0.003 eV higher, because the true minimum falls between two points).
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `wn = (E - E[0]) * EV_TO_CM1` and `zpe = (E[0] - V_min) * EV_TO_CM1`.
# </details>

# %%
r, E, psi = vib_levels(V, mu_CO)
V_min = V(r).min()                     # bottom of the potential (eV)
wn = (E - E[0]) * EV_TO_CM1  # SOLUTION
zpe = (E[0] - V_min) * EV_TO_CM1  # SOLUTION
print(f"zero-point energy = {zpe:.0f} cm⁻¹")
for v in range(1, 5):
    print(f"E({v}) − E(0) = {wn[v]:7.1f} cm⁻¹    E({v}) − E({v-1}) = {wn[v] - wn[v-1]:6.1f} cm⁻¹")
print("experiment: fundamental 2143.3 cm⁻¹")

# %%
lab.check('1.2')

# %% [markdown]
# ## 2. Wavefunctions on the Potential
# **Just run the cell below.** It draws $V(r)$ and, for $v$ = 0–5, each wavefunction offset vertically by its energy, with
# the classical turning points (where $E_v = V$) marked by dots.

# %%
# Just run this cell: wavefunctions drawn at their energies, with classical turning points
plt.figure(figsize=(6.5, 4.8))
plt.plot(r, V(r), "k", lw=2)
for v in range(6):
    sign = np.sign(psi[np.argmax(np.abs(psi[:, v])), v])
    plt.axhline(E[v], color="0.85", lw=0.8)
    plt.plot(r, E[v] + 0.06 * sign * psi[:, v], label=f"v = {v}")
    inside = r[V(r) <= E[v]]
    plt.plot([inside[0], inside[-1]], [E[v], E[v]], "k.", ms=8)
plt.ylim(0, 1.8); plt.xlim(0.95, 1.35)
plt.xlabel("r(C–O) / Å"); plt.ylabel("energy / eV"); plt.legend(loc="upper right", fontsize=8)
plt.title("CO vibrational states on the CCSD(T) curve"); plt.show()

# %% [markdown]
# ### <font color=blue> **Question 2.1 Discussion.** </font>
# <font color=green>
#
# 1. Where is the $v = 0$ wavefunction largest? Where would a *classical* oscillator with the same energy spend most of its time?
# 2. How do the high-$v$ wavefunctions start to look more classical (the correspondence principle)?
# 3. Does any wavefunction extend beyond the turning points? What is that called?
# 4. The wavefunctions lean slightly to longer $r$ as $v$ increases. Which feature of the potential causes this?
# </font>
#
# ---

# %% [markdown] answer
# v = 0 peaks at r_e; a classical oscillator moves slowest, and so spends most time, at the turning points. As v grows, |ψ|² builds up near the turning points, approaching the classical distribution (correspondence principle). Every wavefunction leaks past the turning points into the classically forbidden region (tunnelling). The outer wall of the potential is softer than the inner one, so the wavefunctions lean to longer r and the average bond length grows with v (why the rotational constant B_v decreases with v).

# %% [markdown]
# ---

# %%
lab.check('2.1d')

# %% [markdown]
# ## 3. Anharmonicity

# %% [markdown]
# ### 3.1 The vibrational term formula
# The levels of an anharmonic oscillator fit the term formula
# ([Atkins, Topic 11C](https://global.oup.com/academic/product/atkins-physical-chemistry-9780198769866))
#
# $$G(v) = \omega_e\,(v+\tfrac12) - \omega_e x_e\,(v+\tfrac12)^2,$$
#
# so the spacing shrinks as $v$ increases, and the fundamental is $\tilde\nu_{0\to1} = G(1) - G(0) = \omega_e - 2\,\omega_e x_e$.

# %% [markdown]
# ### <font color=blue> **Question 3.1.** </font>
# `np.polyfit` fits a quadratic $c_2 x^2 + c_1 x + c_0$ in $x = v + \tfrac12$ to your lowest five levels (in cm⁻¹).
# Read off `we` ($\omega_e$) and `wexe` ($\omega_e x_e$) from the coefficients by comparing with $G(v)$.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# The coefficient of $x$ is $\omega_e$, and the coefficient of $x^2$ is $-\omega_e x_e$: `we = c1`, `wexe = -c2`.
# </details>

# %%
x = np.arange(5) + 0.5                       # v + 1/2 for v = 0..4
c2, c1, c0 = np.polyfit(x, E[:5] * EV_TO_CM1, 2)
we = c1  # SOLUTION
wexe = -c2  # SOLUTION
print(f"ω_e      = {we:7.1f} cm⁻¹   (expt 2169.8)")
print(f"ω_e x_e  = {wexe:7.2f} cm⁻¹   (expt 13.29)")
print(f"ω_e − 2ω_e x_e = {we - 2 * wexe:7.1f} cm⁻¹;  your fundamental wn[1] = {wn[1]:.1f} cm⁻¹  (expt 2143.3)")

# %%
lab.check('3.1')

# %% [markdown]
# ## 4. Infrared Intensities: Fundamental and Overtone

# %% [markdown]
# ### 4.1 Transition dipoles
# The strength of an absorption $0\to v$ is proportional to the square of the **transition dipole moment**
#
# $$\langle 0|\mu|v\rangle = \int \psi_0(r)\,\mu(r)\,\psi_v(r)\,dr \;\approx\; \sum_i \psi_0(r_i)\,\mu(r_i)\,\psi_v(r_i)\,h .$$
#
# For a harmonic oscillator with a dipole that is linear in $r$, only $\Delta v = \pm 1$ is allowed. Anharmonicity of the potential
# (*mechanical*) and curvature of $\mu(r)$ (*electrical*) make the **overtone** $0\to 2$ weakly allowed.

# %% [markdown]
# ### <font color=blue> **Question 4.1.** </font>
# Complete the line that computes `M[v]` $= \langle 0|\mu|v\rangle$ (D) for $v$ = 1, 2, 3, and then the intensity ratio of the
# overtone to the fundamental, `ratio_21` $= |\langle 0|\mu|2\rangle|^2 / |\langle 0|\mu|1\rangle|^2$.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# Inside the loop: `M[v] = np.sum(psi[:, 0] * mu_of_r(r) * psi[:, v]) * h`. Then `ratio_21 = M[2]**2 / M[1]**2`.
# </details>

# %%
h = r[1] - r[0]
M = {}
for v in (1, 2, 3):
    M[v] = np.sum(psi[:, 0] * mu_of_r(r) * psi[:, v]) * h  # SOLUTION
ratio_21 = M[2]**2 / M[1]**2  # SOLUTION
for v in (1, 2, 3):
    print(f"|<0|μ|{v}>|² = {M[v]**2:.3e} D²   relative to 0→1: {M[v]**2 / M[1]**2:.2e}")
print(f"overtone / fundamental, including the factor ν̃: {ratio_21 * wn[2] / wn[1]:.2e}  (observed ≈ 1/150 = 6.7e-03)")

# %%
lab.check('4.1')

# %% [markdown]
# ### <font color=blue> **Question 4.2 Discussion.** </font>
# <font color=green>
#
# The integrated absorption of a band is proportional to $\tilde\nu\,|\langle 0|\mu|v\rangle|^2$. The CO overtone near 4260 cm⁻¹ is
# observed to be about 150 times weaker than the fundamental. How does your result compare? Why are overtones weak, and why
# would they vanish for a harmonic oscillator with a linear dipole?
# </font>
#
# ---

# %% [markdown] answer
# The calculation gives the overtone about 210 times weaker than the fundamental (|M₀₂|²/|M₀₁|² = 2.4 × 10⁻³, times ν̃₀₂/ν̃₀₁ ≈ 2), the right order of magnitude (observed ≈ 1/150). For a harmonic oscillator, μ(r) linear in r connects only neighbouring levels, so ⟨0|μ|2⟩ = 0 exactly. Overtones need either an anharmonic potential (mechanical anharmonicity) or a curved dipole function (electrical anharmonicity); both are small near r_e, so the overtone is down by about two orders of magnitude. The exact ratio is sensitive to the B3LYP dipole curvature, because the two contributions partly cancel.

# %% [markdown]
# ---

# %%
lab.check('4.2d')

# %% [markdown]
# ## 5. The Isotope Shift
# In the Born–Oppenheimer approximation the potential depends only on the electrons and nuclear charges, so **isotopes have the
# same potential** and differ only in the reduced mass. The harmonic estimate is $\tilde\nu \propto \mu^{-1/2}$.

# %% [markdown]
# ### <font color=blue> **Question 5.1.** </font>
# Compute the reduced mass `mu13` of ¹³C¹⁶O (mass of ¹³C: 13.003355 amu), its levels on the **same** potential `V`, and its
# fundamental `f13` in cm⁻¹. Experiment: 2096.1 cm⁻¹, a shift of −47.2 cm⁻¹.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `mu13 = 13.003355 * m_O / (13.003355 + m_O)`; `_, E13, _ = vib_levels(V, mu13)` is given; `f13 = (E13[1] - E13[0]) * EV_TO_CM1`.
# </details>

# %%
mu13 = 13.003355 * m_O / (13.003355 + m_O)  # SOLUTION
_, E13, _ = vib_levels(V, mu13)
f13 = (E13[1] - E13[0]) * EV_TO_CM1  # SOLUTION
print(f"¹³C¹⁶O fundamental = {f13:.1f} cm⁻¹, shift = {f13 - wn[1]:.1f} cm⁻¹   (expt 2096.1, −47.2)")
print(f"harmonic estimate of the shift: {wn[1] * (np.sqrt(mu_CO / mu13) - 1):.1f} cm⁻¹")

# %%
lab.check('5.1')

# %% [markdown]
# ### 📋 Summary of your results
# **Just run the cell below.**

# %%
# Just run this cell: summary table
summary = Table().with_columns(
    'Quantity', ['zero-point energy (cm⁻¹)', 'fundamental 0→1 (cm⁻¹)', 'ω_e (cm⁻¹)', 'ω_e x_e (cm⁻¹)',
                 'overtone/fundamental intensity', '¹³C¹⁶O fundamental (cm⁻¹)', '¹³C isotope shift (cm⁻¹)'],
    'This lab', [round(zpe), round(wn[1], 1), round(we, 1), round(wexe, 2), f"{ratio_21 * wn[2] / wn[1]:.1e}",
                 round(f13, 1), round(f13 - wn[1], 1)],
    'Experiment', ['1081.6', '2143.3', '2169.8', '13.29', '≈ 6.7e-03', '2096.1', '−47.2'])
summary.show()

# %% [markdown]
# ### Looking ahead
# This exact grid solution is easy in one dimension. A triatomic molecule has $3N - 5 = 4$ vibrations (linear), and a grid of
# 600 points per coordinate would have $600^4 \approx 10^{11}$ points. In Labs 4–6 you will instead use (1) the **harmonic**
# normal-mode approximation and (2) **classical molecular dynamics**, which costs little in many dimensions but has no
# zero-point energy. Keep the zero-point energy from this lab in mind for that comparison.

# %% [markdown]
# ### 📝 Quick Check — Vibrational Levels
# Test what you've learned. This is just for practice — answer as many times as you like, it isn't recorded.

# %%
with open("questionsComp03.json", "r") as f:
    display_quiz(json.load(f), shuffle_answers=True)

# %% [markdown]
# ---
# ## 6. Conclusions

# %% [markdown]
# ### <font color=blue> **Question 6. Conclusions.** </font>
# <div style="border:2px solid #1f4e79; border-radius:8px; padding:10px 14px; background:#f3f7fb; color:#1a1a1a;">
#
# Write the **Conclusions** of this lab in the cell below (one or two paragraphs, about 150–300 words), using **your own numbers**
# from the summary table. Address:
#
# 1. How you know the numerical method is right (the harmonic test), and how close your fundamental is to experiment.
# 2. What anharmonicity does to the levels, and why the fundamental lies below ω_e.
# 3. Why overtones are weak, and what the isotope shift confirms about the Born–Oppenheimer approximation.
# 4. Which remaining error you would attack next: the potential, the dipole curve, or the method?
# </div>
#
# ---

# %% [markdown] answer
# The finite-difference method reproduces the harmonic-oscillator levels to better than 0.1 %, so the remaining differences from experiment come from the potential, not the numerics. On the CCSD(T)/cc-pVTZ curve the CO fundamental is 2145 cm⁻¹, within 2 cm⁻¹ of the observed 2143.3 cm⁻¹. Anharmonicity makes the levels converge: the spacing drops by 26 cm⁻¹ per quantum (ω_e x_e = 13.0 cm⁻¹), so the fundamental lies 2ω_e x_e below ω_e. The overtone is about 210 times weaker than the fundamental (observed ≈ 150), the right order of magnitude, because it is allowed only through the anharmonicity of the potential and the curvature of the dipole. The ¹³C shift (−47.3 cm⁻¹ vs −47.2 cm⁻¹ observed) confirms that isotopes share one potential and differ only in mass. The overtone intensity is the weakest result, because it depends on the second derivative of the B3LYP dipole; a CCSD dipole curve would be the next improvement.

# %% [markdown]
# ---

# %% [markdown]
# ## 7. Reflection

# %% [markdown]
# ### <font color=blue> **Question 7.** </font>
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
# (Instructor version: students write their reflection here.) Building the Hamiltonian as a matrix made the eigenvalue idea concrete.

# %% [markdown]
# ---

# %% [markdown]
# #### Save notebook before check
# This cell checks your **Abstract** (at the top of the notebook), your **Conclusions** and your **Reflection**.

# %%
lab.check('abstract')
lab.check('6c')
lab.check('7d')

# %% [markdown]
# ## 📚 References
# 1. P. Atkins, J. de Paula and J. Keeler, *Atkins' Physical Chemistry*, 11th ed., Oxford University Press (2018): Topic 7E (the harmonic oscillator), Topic 11C (vibrational spectroscopy of diatomic molecules: anharmonicity, overtones, the Birge–Sponer plot).
# 2. D. A. McQuarrie and J. D. Simon, *Physical Chemistry: A Molecular Approach*, University Science Books (1997), Ch. 5 (harmonic oscillator) and 13 (molecular spectroscopy).
# 3. D. T. Colbert and W. H. Miller, "A novel discrete variable representation for quantum mechanical reactive scattering via the S-matrix Kohn method," *J. Chem. Phys.* **96**, 1982 (1992), [doi:10.1063/1.462100](https://doi.org/10.1063/1.462100) (grid methods for the vibrational Schrödinger equation).
# 4. K. P. Huber and G. Herzberg, *Molecular Spectra and Molecular Structure IV. Constants of Diatomic Molecules*, Van Nostrand Reinhold (1979); values via the [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/).
# 5. C. Chackerian Jr. and R. H. Tipping, "Vibration–rotational and rotational intensities for CO isotopes," *J. Mol. Spectrosc.* **99**, 431 (1983), [doi:10.1016/0022-2852(83)90324-6](https://doi.org/10.1016/0022-2852(83)90324-6).
# 6. E. Tiesinga et al., "CODATA Recommended Values of the Fundamental Physical Constants: 2018," *Rev. Mod. Phys.* **93**, 025010 (2021).

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
lab.submit(name, lab_id='Comp03')
