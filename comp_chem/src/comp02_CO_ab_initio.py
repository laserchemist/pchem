# LAB
# %% [markdown]
# # Computational Lab 2: Ab Initio Theory and Basis Sets — Carbon Monoxide
# *CHEM 3376 — Physical Chemistry Laboratory: Data Analysis & Computation*
#
# In Lab 1 you saw that Hartree–Fock misses the **correlation energy**, and that full CI, which recovers all of it, is
# impossible beyond a few electrons. DFT gave good results for O₂, but there is no systematic way to improve a functional.
# **Ab initio** ("from the beginning") wavefunction methods form a *hierarchy*: each level adds more of the physics the
# previous one left out, at a higher cost. In this lab you climb that hierarchy for carbon monoxide and watch the results
# converge toward experiment:
#
# 1. the **energy** of CO with Hartree–Fock, MP2, CCSD and CCSD(T), and the **correlation energy** each recovers;
# 2. **potential energy curves**, bond lengths and vibrational wavenumbers from each method;
# 3. **basis-set convergence**: double-, triple- (and quadruple-) zeta basis sets;
# 4. the **dipole moment** of CO, a famously delicate property, and its change with bond length, which makes CO absorb
#    infrared light.
#
# CO is a good test: it has a triple bond, its stretch is a strong IR absorber (2143 cm⁻¹), and its dipole moment (0.11 D)
# is tiny and points the "wrong" way. The curve you compute here is the starting point for Lab 3.
#
# Most answers are one line. Stuck? Open the 💡 **Hint** under each question. Cells marked *"Just run this cell"* hold
# code you don't need to read. Some cells take a minute: a `[*]` next to a cell means it is still running.
#
# <br>**<center>Learning Goals**
#
# |Area|Concept|
# |---|---|
# |Electronic structure|The ab initio hierarchy: Hartree–Fock, MP2, CCSD, CCSD(T); correlation energy; cost scaling with basis size|
# |Basis sets|Correlation-consistent basis sets cc-pVXZ; convergence with the cardinal number X|
# |Spectroscopy|Bond length, force constant and harmonic wavenumber from a computed curve; why CO absorbs infrared light|
# |Molecular properties|The dipole moment and its derivative dμ/dr; when Hartree–Fock gets the sign wrong|
# |Scientific practice|Agreement with experiment is not the same as convergence|
# |Python|`if`/`elif`, dictionaries, `np.polyfit`, a function that returns several values|

# %% [markdown]
# ---
# ## 📄 Abstract
#
# <div style="border:2px solid #1f4e79; border-radius:8px; padding:10px 14px; background:#f3f7fb; color:#1a1a1a;">
#
# **Write this section last**, after you have finished the lab. Your abstract goes in the cell below, replacing `ANSWER`.
#
# In **150–250 words**, as in a journal article: state the purpose of the calculations; name the methods (HF, MP2, CCSD(T),
# B3LYP, the basis sets, PySCF); give your **key numbers** with experiment beside them (r_e and ν̃e of CO by method, the
# basis-set trend, the dipole moment and dμ/dr); and finish with the main conclusion.
# </div>

# %% [markdown] answer
# The ab initio hierarchy Hartree–Fock → MP2 → CCSD → CCSD(T) was applied to carbon monoxide with PySCF to see how electron correlation and basis-set size control computed molecular properties. At r = 1.128 Å with cc-pVDZ the correlation energy is −7.9 eV (MP2), −8.1 eV (CCSD) and −8.4 eV (CCSD(T)), only 0.3 % of the total energy but larger than the bond-length dependence that sets the vibrational frequency. Potential-energy curves in cc-pVTZ gave r_e = 1.105, 1.135 and 1.133 Å and ν̃e = 2425, 2138 and 2171 cm⁻¹ for HF, MP2 and CCSD(T) (experiment 1.1283 Å, 2169.8 cm⁻¹): Hartree–Fock gives a bond that is too short and too stiff because it omits the mixing of antibonding configurations, MP2 overcorrects, and CCSD(T) agrees best. CCSD(T)/cc-pVDZ gave 1.144 Å and 2146 cm⁻¹, so the agreement in cc-pVTZ is partly a cancellation of remaining basis-set and core-correlation errors. Hartree–Fock gives the wrong sign of the dipole moment (−0.24 D, C⁺O⁻), whereas B3LYP (+0.12 D) and CCSD (+0.18 D) give the observed C⁻O⁺ polarity. The B3LYP dipole derivative dμ/dr = −3.5 D Å⁻¹ explains why the CO stretch absorbs strongly even though μ itself is tiny. Systematic improvement of both method and basis set is needed before agreement with experiment can be trusted.

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
ver = '2026Comp02V1'

# %% [markdown]
# ### 🧰 Standard imports
# `numpy`, `matplotlib` and five parts of PySCF: `gto` (molecules and basis sets), `scf` (Hartree–Fock), `mp` (MP2),
# `cc` (coupled cluster) and `dft`. `labkit` is this course's small helper package (unit constants, atomic masses,
# a 3-D viewer); it lives in the folder above this one.
# The last lines load this lab's **answer checks**: after each question, run `lab.check(...)`. A ✅ means you got it;
# a ❌ tells you what to fix. Fix the cell and run the check again, as often as you like: there is no penalty for trying.

# %%
# Standard imports for this lab
import time
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from pyscf import gto, scf, mp, cc, dft
for _up in ("..", os.path.join("..", "..")):          # labkit is in comp_chem/, the folder above this lab
    if os.path.isdir(os.path.join(_up, "labkit")):
        sys.path.insert(0, os.path.abspath(_up))
        break
import labkit as lk

# Answer checks for this lab: the checker (pchem_check.py) and the answer key (answers_comp02.json) are in this folder
from pchem_check import Checker
lab = Checker('comp02', 'answers_comp02.json')

# %% [markdown]
# Now run the cell below to enter your name.

# %%
# Enter your name as a string, e.g. name = 'Linus Pauling'
name = 'Instructor'  # SOLUTION

# %% [markdown]
# Constants from `labkit` (CODATA 2018). Lengths are in Å, energies in eV, masses in amu.

# %%
HARTREE_TO_EV = lk.HARTREE_TO_EV      # eV per hartree
m_C, m_O = lk.MASS["C"], lk.MASS["O"]  # amu, ¹²C and ¹⁶O
mu_CO = m_C * m_O / (m_C + m_O)       # reduced mass, amu
print(f"reduced mass of CO = {mu_CO:.4f} amu")
lk.show_molecule(["C", "O"], np.array([[0, 0, 0], [0, 0, 1.128]]), labels=True)

# %% [markdown]
# ## 1. Climbing the Ladder at One Geometry

# %% [markdown]
# ### 1.1 The ab initio hierarchy
# Hartree–Fock puts each electron in an orbital and lets it feel only the *average* repulsion of the others. Real electrons
# also avoid each other instant by instant; the energy that this **correlation** lowers is the **correlation energy**
# ([Atkins, Focus 9, computational chemistry](https://global.oup.com/academic/product/atkins-physical-chemistry-9780198769866);
# [McQuarrie & Simon, Ch. 11](https://uscibooks.aip.org/books/physical-chemistry-a-molecular-approach/)):
#
# $$ E_\text{corr} = E_\text{exact} - E_\text{HF} \quad(<0). $$
#
# The correlated methods start from the Hartree–Fock orbitals and mix in **excited configurations**, in which electrons
# have been promoted into empty (virtual) orbitals:
#
# | method | what it includes | cost grows as |
# |---|---|---|
# | Hartree–Fock (HF) | each electron feels the *average* field of the others | $N^4$ |
# | MP2 | pairs of electrons avoiding each other, 2nd-order perturbation theory | $N^5$ |
# | CCSD | single and double excitations to all orders (coupled cluster) | $N^6$ |
# | CCSD(T) | CCSD plus an estimate of triples: "the gold standard" | $N^7$ |
#
# Here $N$ measures the size of the basis set. Doubling the basis multiplies the CCSD(T) cost by $2^7 = 128$.
#
# In PySCF each correlated method starts from a converged HF calculation:
#
# ```python
# mf = scf.RHF(mol).run()                  # Hartree–Fock
# e_mp2 = mp.MP2(mf).run().e_tot           # MP2 total energy (hartree)
# mycc = cc.CCSD(mf).run()                 # CCSD; its energy is mycc.e_tot
# e_ccsd_t = mycc.e_tot + mycc.ccsd_t()    # CCSD(T) = CCSD + the triples correction
# ```

# %% [markdown]
# ### <font color=blue> **Question 1.1.** </font>
# Complete `co_energy(r, method, basis)`, which returns the energy of CO in **eV** for `method` = `"hf"`, `"mp2"`,
# `"ccsd"` or `"ccsd(t)"`. Two lines are missing: the MP2 energy, and the triples correction for CCSD(T).
#
# 🐍 **Python tip — `if` / `elif` / `else`.** Python tries each condition in order and runs the block under the first one
# that is true; `else` catches everything left. The indentation shows which lines belong to which block.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `e = mp.MP2(mf).run().e_tot` and `e = e + mycc.ccsd_t()`. Both are in hartree; the last line converts to eV.
# </details>

# %%
def co_energy(r, method="ccsd(t)", basis="cc-pvdz"):
    """Total electronic energy (eV) of CO at bond length r (Å)."""
    mol = gto.M(atom=[["C", (0, 0, 0)], ["O", (0, 0, r)]], basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    if method == "hf":
        e = mf.e_tot
    elif method == "mp2":
        e = mp.MP2(mf).run().e_tot  # SOLUTION
    else:
        mycc = cc.CCSD(mf).run()
        e = mycc.e_tot
        if method == "ccsd(t)":
            e = e + mycc.ccsd_t()  # SOLUTION
    return e * HARTREE_TO_EV

print(f"HF/STO-3G energy of CO at 1.128 Å: {co_energy(1.128, 'hf', 'sto-3g'):.3f} eV")

# %%
lab.check('1.1')

# %% [markdown]
# ### <font color=blue> **Question 1.2.** </font>
# The cell below computes the energy with each method at r = 1.128 Å in cc-pVDZ, and times it.
# Complete the line that computes the **correlation energy** of each method, `E_corr[m]`, in eV.
#
# 🐍 **Python tip — dictionaries.** `E = {}` makes an empty dictionary; `E["mp2"] = -3000.1` stores a value under the
# key `"mp2"`, and `E["mp2"]` reads it back. Here the keys are method names.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# The correlation energy is the method's energy minus the Hartree–Fock energy: `E_corr[m] = E[m] - E["hf"]`.
# </details>

# %%
methods = ["hf", "mp2", "ccsd", "ccsd(t)"]
E, E_corr, cpu = {}, {}, {}
for m in methods:
    t = time.time()
    E[m] = co_energy(1.128, m)
    cpu[m] = time.time() - t
for m in methods:
    E_corr[m] = E[m] - E["hf"]  # SOLUTION
    print(f"{m:8s} E = {E[m]:12.4f} eV   E_corr = {E_corr[m]:7.3f} eV   ({cpu[m]:.2f} s)")

# %%
lab.check('1.2')

# %% [markdown]
# ### <font color=blue> **Question 1.3 Discussion.** </font>
# <font color=green>
#
# 1. The correlation energy is a few eV, only about 0.3 % of the total energy. Why does it still matter for chemistry?
#    Compare it with the CO bond energy (11.1 eV) and with one vibrational quantum of CO (0.27 eV).
# 2. Is the order of the correlation energies (MP2, CCSD, CCSD(T)) what you expected from the table? Is a more negative
#    correlation energy always "better"? (Think about the variational principle from Lab 1: are these methods variational?)
# </font>
#
# ---

# %% [markdown] answer
# Most of the total energy belongs to core electrons that do not change in chemistry. What matters is how E_corr *changes* with geometry: tenths of an eV over a bond stretch, comparable to a vibrational quantum, so it changes the bond length and curvature (frequency), and much of the bond energy. MP2 recovers most of the correlation energy, CCSD a little more and (T) more again. But MP2 and coupled cluster are not variational, so a lower energy is not automatically closer to the exact one; the test is convergence of properties toward experiment.

# %% [markdown]
# ---

# %%
lab.check('1.3d')

# %% [markdown]
# ## 2. Potential Energy Curves and Spectroscopic Constants

# %% [markdown]
# ### 2.1 From a curve to a vibrational wavenumber
# Near the minimum, the curve is fitted by a polynomial in $x = r - r_0$. Its minimum gives the bond length $r_e$, and its
# **curvature** there is the force constant $k = d^2E/dr^2$. As in Lab 1, the harmonic wavenumber is
#
# $$ \tilde\nu_e = \frac{1}{2\pi c}\sqrt{\frac{k}{\mu}} . $$
#
# With $k$ in eV/Å² and $\mu$ in amu, $\sqrt{k/\mu}$ needs a units factor: $\omega\,[\text{rad/fs}] = \sqrt{(k/\mu)\times\texttt{lk.ACC}}$,
# and $c$ = `lk.C_CM_PER_FS` = 2.998 × 10⁻⁵ cm/fs.

# %% [markdown]
# ### <font color=blue> **Question 2.1.** </font>
# Complete `fit_minimum(r, E, mu_amu)`. The fit and the minimum are done for you; write the two missing lines:
# - `k`, the second derivative of the polynomial `p` at the minimum `x` (eV/Å²);
# - `wn`, the harmonic wavenumber in cm⁻¹.
#
# The cell tests your function on a made-up harmonic curve, $E = \tfrac12 (120\ \text{eV/Å}^2)(r - 1.13\ \text{Å})^2$,
# whose answers you know: $r_e$ = 1.13 Å and $k$ = 120 eV/Å².
#
# 🐍 **Python tip — polynomials.** `np.polyfit(x, y, 4)` returns the coefficients `p` of the best 4th-order polynomial;
# `np.polyder(p, 2)` gives the coefficients of its second derivative, and `np.polyval(q, x)` evaluates a polynomial `q` at `x`.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `k = np.polyval(np.polyder(p, 2), x)` and `wn = np.sqrt(k / mu_amu * lk.ACC) / (2 * np.pi * lk.C_CM_PER_FS)`.
# </details>

# %%
def fit_minimum(r, E, mu_amu):
    """Return (r_e in Å, k in eV/Å², wavenumber in cm⁻¹) from energies E (eV) at bond lengths r (Å) near a minimum."""
    r0 = r[np.argmin(E)]                          # the lowest computed point
    p = np.polyfit(r - r0, E, 4)                  # E ≈ polynomial in x = r - r0
    roots = np.roots(np.polyder(p))               # where dE/dx = 0
    x = roots[np.argmin(np.abs(roots))].real      # the root closest to r0 is the minimum
    k = np.polyval(np.polyder(p, 2), x)  # SOLUTION
    wn = np.sqrt(k / mu_amu * lk.ACC) / (2 * np.pi * lk.C_CM_PER_FS)  # SOLUTION
    return r0 + x, k, wn

r_test = np.linspace(1.08, 1.18, 7)
print("test curve: r_e = %.4f Å, k = %.1f eV/Å², wn = %.1f cm⁻¹" % fit_minimum(r_test, 0.5 * 120 * (r_test - 1.13)**2, mu_CO))

# %%
lab.check('2.1')

# %% [markdown]
# ### <font color=blue> **Question 2.2.** </font>
# Now the real curves. The cell computes HF, MP2 and CCSD(T) energies in **cc-pVTZ** at 7 bond lengths from 1.08 to 1.18 Å
# (about a minute) and fits each. Complete the two lines inside the loop:
# - `E_curve`, an array of `co_energy(r, m, "cc-pvtz")` for every `r` in `r_pts`;
# - `results[m]`, the fit of that curve.
#
# 🐍 **Python tip — list comprehensions.** `[f(r) for r in r_pts]` makes a list of `f(r)` for every `r`;
# wrap it in `np.array(...)` to do arithmetic on it.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `E_curve = np.array([co_energy(r, m, "cc-pvtz") for r in r_pts])` and `results[m] = fit_minimum(r_pts, E_curve, mu_CO)`.
# </details>

# %%
r_pts = np.linspace(1.08, 1.18, 7)
results, curves = {}, {}
t = time.time()
for m in ("hf", "mp2", "ccsd(t)"):
    E_curve = np.array([co_energy(r, m, "cc-pvtz") for r in r_pts])  # SOLUTION
    curves[m] = E_curve
    results[m] = fit_minimum(r_pts, E_curve, mu_CO)  # SOLUTION
print(f"({time.time() - t:.0f} s)\n")
print(f"{'method':10s} {'r_e / Å':>9s} {'k / eV Å⁻²':>12s} {'ν̃e / cm⁻¹':>11s}")
for m, (re, k, wn) in results.items():
    print(f"{m:10s} {re:9.4f} {k:12.1f} {wn:11.0f}")
print(f"{'experiment':10s} {1.1283:9.4f} {'':12s} {2169.8:11.0f}")

# %%
lab.check('2.2')

# %% [markdown]
# **Just run the cell below.** It plots the three curves, each shifted so that its minimum is zero, so you can compare
# their *shapes*.

# %%
# Just run this cell: the three curves near the minimum
r_fine = np.linspace(1.08, 1.18, 200)
for m, col in zip(("hf", "mp2", "ccsd(t)"), ("C0", "C1", "C2")):
    p = np.polyfit(r_pts, curves[m], 4)
    shift = np.polyval(p, r_fine).min()
    plt.plot(r_pts, curves[m] - shift, "o", color=col)
    plt.plot(r_fine, np.polyval(p, r_fine) - shift, color=col, label=f"{m.upper()} / cc-pVTZ")
plt.axvline(1.1283, color="k", ls="--", lw=1, label="experiment r_e")
plt.xlabel("r(C–O) / Å"); plt.ylabel("E − E_min / eV"); plt.legend(); plt.show()

# %% [markdown]
# ### <font color=blue> **Question 2.3 Discussion.** </font>
# <font color=green>
#
# 1. Hartree–Fock gives a bond that is too short and too stiff. Explain this using the idea that HF does not let electrons
#    avoid each other: correlation mixes in configurations with electrons promoted into the π* and σ* **antibonding** orbitals.
#    What does that do to the bond?
# 2. Does MP2 or CCSD(T) agree better with experiment? Which method would you use for a molecule twice the size of CO, and why?
# </font>
#
# ---

# %% [markdown] answer
# Mixing in configurations with electrons in π*/σ* antibonding orbitals lets the electrons avoid each other, but it also weakens the bond: the bond gets longer and softer. Hartree–Fock leaves this out, so it overbinds near r_e: r_e = 1.105 Å (0.023 Å short) and ν̃e = 2425 cm⁻¹ (12 % high). MP2 overcorrects for CO (1.135 Å, 2138 cm⁻¹); CCSD(T) is closest (1.133 Å, 2171 cm⁻¹). For a molecule twice the size CCSD(T) costs 2⁷ = 128 times more, so MP2 or DFT would be the practical choice, checked against CCSD(T) on a smaller model.

# %% [markdown]
# ---

# %%
lab.check('2.3d')

# %% [markdown]
# ## 3. Basis-Set Convergence

# %% [markdown]
# ### 3.1 Correlation-consistent basis sets
# The answer depends on the **basis set** as well as the method. Dunning's correlation-consistent sets cc-pVXZ
# (X = D, T, Q for double, triple, quadruple zeta) add a full shell of functions at each step, so that the correlation energy
# converges smoothly, roughly as $X^{-3}$ (Dunning 1989). A result can only be trusted when it has stopped changing with X.

# %% [markdown]
# ### <font color=blue> **Question 3.1.** </font>
# You already have the CCSD(T)/cc-pVTZ fit. Complete the line that computes the CCSD(T) curve in **cc-pVDZ** at the same
# points, and fit it. If you have time (about 3 minutes), add `"cc-pvqz"` to `extra_bases` and run the cell again.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# Same as Question 2.2 with a different basis: `E_dz = np.array([co_energy(r, "ccsd(t)", "cc-pvdz") for r in r_pts])`.
# </details>

# %%
conv = {"cc-pvtz": results["ccsd(t)"]}
E_dz = np.array([co_energy(r, "ccsd(t)", "cc-pvdz") for r in r_pts])  # SOLUTION
conv["cc-pvdz"] = fit_minimum(r_pts, E_dz, mu_CO)

extra_bases = []                    # e.g. ["cc-pvqz"]: about 20 s per point
for b in extra_bases:
    conv[b] = fit_minimum(r_pts, np.array([co_energy(r, "ccsd(t)", b) for r in r_pts]), mu_CO)

X_card = {"cc-pvdz": 2, "cc-pvtz": 3, "cc-pvqz": 4}
order = sorted(conv, key=X_card.get)
for b in order:
    print(f"CCSD(T)/{b}: r_e = {conv[b][0]:.4f} Å   ν̃e = {conv[b][2]:.0f} cm⁻¹")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot([X_card[b] for b in order], [conv[b][0] for b in order], "o-"); ax[0].axhline(1.1283, color="k", ls="--")
ax[1].plot([X_card[b] for b in order], [conv[b][2] for b in order], "o-"); ax[1].axhline(2169.8, color="k", ls="--", label="experiment")
for a, lab_y in zip(ax, ("r_e / Å", "ν̃e / cm⁻¹")):
    a.set_xticks([2, 3, 4], ["DZ", "TZ", "QZ"]); a.set_ylabel(lab_y); a.set_xlabel("basis set")
ax[1].legend(); plt.tight_layout(); plt.show()

# %%
lab.check('3.1')

# %% [markdown]
# ### <font color=blue> **Question 3.2 Discussion.** </font>
# <font color=green>
#
# CCSD(T)/cc-pVTZ agrees with the experimental ν̃e almost perfectly. Is the calculation *converged*? Use your DZ → TZ (→ QZ)
# trend to argue whether the next basis set would move the answer toward experiment or past it. What is the difference
# between *agreement* and *convergence*?
# </font>
#
# ---

# %% [markdown] answer
# No: going from DZ to TZ shortened r_e by about 0.011 Å and raised ν̃e by about 25 cm⁻¹, and the trend continues to QZ (r_e is still 0.004 Å long in TZ; core correlation, left out here, shortens it further). The near-perfect TZ frequency is partly a cancellation of the remaining basis-set error against other small errors (core correlation, higher excitations). Agreement means one number matches experiment; convergence means the answer has stopped changing as the method and basis improve. Only a converged result can be trusted for a molecule where there is no experiment.

# %% [markdown]
# ---

# %%
lab.check('3.2d')

# %% [markdown]
# ## 4. The Dipole Moment, and Why CO Absorbs Infrared Light

# %% [markdown]
# ### 4.1 A delicate property
# Unlike O₂, CO is **heteronuclear**, so its dipole moment is not zero by symmetry. Experimentally $\mu$ = 0.11 D with the
# **negative end on carbon** (C⁻O⁺), the opposite of what electronegativity suggests: charge transfer toward O is almost
# exactly cancelled by the carbon lone pair, which points away from the bond.
#
# PySCF reports the dipole vector $\boldsymbol\mu = \sum_A Z_A \mathbf R_A - \int \rho(\mathbf r)\,\mathbf r\, d\mathbf r$, which points
# from − to +. With C at the origin and O on +z, **C⁻O⁺ means $\mu_z > 0$**.
#
# For a correlated method, the dipole comes from that method's electron density (its *density matrix*):
# `dm = mycc.make_rdm1(ao_repr=True)`, then `mf.dip_moment(dm=dm, unit="Debye", verbose=0)`.

# %% [markdown]
# **Just run the cell below** (about 30 s). It computes $\mu_z$ at r = 1.128 Å with HF, B3LYP and CCSD in cc-pVTZ.

# %%
# Just run this cell: the dipole moment of CO with three methods
def co_dipole(r, method="b3lyp", basis="cc-pvtz"):
    """z component of the CO dipole (debye); C at the origin, O on +z."""
    mol = gto.M(atom=[["C", (0, 0, 0)], ["O", (0, 0, r)]], basis=basis, verbose=0)
    if method == "b3lyp":
        mf = dft.RKS(mol, xc="b3lyp").run()
        return mf.dip_moment(unit="Debye", verbose=0)[2]
    mf = scf.RHF(mol).run()
    if method == "hf":
        return mf.dip_moment(unit="Debye", verbose=0)[2]
    mycc = cc.CCSD(mf).run()
    return mf.dip_moment(dm=mycc.make_rdm1(ao_repr=True), unit="Debye", verbose=0)[2]

mu_z = {m: co_dipole(1.128, m) for m in ("hf", "b3lyp", "ccsd")}
for m, v in mu_z.items():
    print(f"{m:6s} μ_z = {v:+.3f} D   ({'C⁻O⁺' if v > 0 else 'C⁺O⁻'})")
print("expt   μ_z = +0.110 D   (C⁻O⁺)")

# %% [markdown]
# ### 4.2 Infrared intensity
# A vibration absorbs infrared light only if the dipole moment **changes** as the molecule vibrates. The intensity of the
# fundamental is proportional to $(d\mu/dr)^2$ at $r_e$
# ([Atkins, Topic 11C, the gross selection rule](https://global.oup.com/academic/product/atkins-physical-chemistry-9780198769866)).

# %% [markdown]
# ### <font color=blue> **Question 4.2.** </font>
# With B3LYP, compute $\mu_z$ at 1.10 and 1.16 Å, and estimate $d\mu/dr$ (D/Å) by a **central difference**:
#
# $$ \frac{d\mu}{dr} \approx \frac{\mu(1.16) - \mu(1.10)}{1.16 - 1.10} . $$
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `mu_plus = co_dipole(1.16)`, `mu_minus = co_dipole(1.10)` and `dmu_dr = (mu_plus - mu_minus) / (1.16 - 1.10)`.
# </details>

# %%
mu_plus = co_dipole(1.16)  # SOLUTION
mu_minus = co_dipole(1.10)  # SOLUTION
dmu_dr = (mu_plus - mu_minus) / (1.16 - 1.10)  # SOLUTION
print(f"B3LYP: μ_z(1.10) = {mu_minus:+.3f} D, μ_z(1.16) = {mu_plus:+.3f} D,  dμ/dr = {dmu_dr:.2f} D/Å")

# %%
lab.check('4.2')

# %% [markdown]
# ### <font color=blue> **Question 4.3 Discussion.** </font>
# <font color=green>
#
# 1. Which methods get the **sign** of the dipole right? What does it tell you that Hartree–Fock gets even the direction wrong
#    for such a small property?
# 2. The *value* of μ at $r_e$ is tiny, yet CO absorbs strongly at 2143 cm⁻¹. Which quantity controls the IR intensity?
#    Compare with O₂ in Lab 1.
# </font>
#
# ---

# %% [markdown] answer
# HF gives −0.24 D (C⁺O⁻), the wrong sign; B3LYP (+0.12 D) and CCSD (+0.18 D) give the observed C⁻O⁺. The dipole is a small difference between two large, opposing effects (charge transfer toward O and the carbon lone pair), so a modest error in either flips the sign; small properties are a severe test of a method. The intensity depends on (dμ/dr)², not on μ: |dμ/dr| ≈ 3.5 D/Å is large, so the band is strong. O₂ has μ = 0 at every bond length, so dμ/dr = 0 and its stretch is IR-inactive.

# %% [markdown]
# ---

# %%
lab.check('4.3d')

# %% [markdown]
# ## 5. Save a Curve for Lab 3
# **Just run the cell below** (about a minute). Lab 3 solves the vibrational Schrödinger equation on the CCSD(T) curve, over
# a wider range of bond lengths, with the B3LYP dipole curve for the intensities. The file goes into `comp_chem/results/`,
# where Lab 3 looks for it.

# %%
# Just run this cell: the CCSD(T)/cc-pVTZ curve and B3LYP dipole curve from 0.90 to 1.60 Å, saved for Lab 3
r_wide = np.linspace(0.90, 1.60, 29)
t = time.time()
E_wide = np.array([co_energy(r, "ccsd(t)", "cc-pvtz") for r in r_wide])
mu_wide = np.array([co_dipole(r, "b3lyp") for r in r_wide])
np.savez(lk.results_path("co_ccsdt_curve.npz"), r=r_wide, E_eV=E_wide - E_wide.min(), mu_D=mu_wide)
print(f"saved results/co_ccsdt_curve.npz ({time.time() - t:.0f} s)")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(r_wide, E_wide - E_wide.min(), "o-"); ax[0].set_ylabel("E − E_min / eV")
ax[1].plot(r_wide, mu_wide, "o-"); ax[1].set_ylabel("μ_z / D")
for a in ax:
    a.set_xlabel("r(C–O) / Å")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 📋 Summary of your results
# **Just run the cell below.** It collects your results in one table for your abstract.

# %%
# Just run this cell: summary table
rows = [("HF / cc-pVTZ", results["hf"]), ("MP2 / cc-pVTZ", results["mp2"]),
        ("CCSD(T) / cc-pVDZ", conv["cc-pvdz"]), ("CCSD(T) / cc-pVTZ", results["ccsd(t)"])]
rows += [(f"CCSD(T) / {b}", conv[b]) for b in extra_bases]
summary = Table().with_columns(
    'Method / basis', [n for n, _ in rows] + ['experiment'],
    'r_e (Å)', [round(v[0], 4) for _, v in rows] + [1.1283],
    'k (eV/Å²)', [round(v[1], 1) for _, v in rows] + ['—'],
    'ν̃e (cm⁻¹)', [round(v[2]) for _, v in rows] + [2169.8])
summary.show()
print(f"E_corr at 1.128 Å (cc-pVDZ): MP2 {E_corr['mp2']:.2f}, CCSD {E_corr['ccsd']:.2f}, CCSD(T) {E_corr['ccsd(t)']:.2f} eV")
print(f"μ_z at 1.128 Å (cc-pVTZ): HF {mu_z['hf']:+.3f}, B3LYP {mu_z['b3lyp']:+.3f}, CCSD {mu_z['ccsd']:+.3f} D;  B3LYP dμ/dr = {dmu_dr:.2f} D/Å")

# %% [markdown]
# ### 📝 Quick Check — The Ab Initio Ladder
# Test what you've learned. This is just for practice — answer as many times as you like, it isn't recorded.

# %%
with open("questionsComp02.json", "r") as f:
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
# 1. How each step up the ladder (HF → MP2 → CCSD(T)) changed r_e and ν̃e, and why.
# 2. What the basis-set trend says about how far your best result is from convergence.
# 3. What the dipole moment taught you about judging a method by one property.
# 4. Which result you trust most, and what you would calculate next to test it.
# </div>
#
# ---

# %% [markdown] answer
# Electron correlation lengthens and softens the CO bond: Hartree–Fock gives 1.105 Å and 2425 cm⁻¹, far from experiment (1.1283 Å, 2169.8 cm⁻¹), because it leaves out the mixing of π*/σ* configurations that lets the electrons avoid each other. MP2 overcorrects (1.135 Å, 2138 cm⁻¹) and CCSD(T) is closest (1.133 Å, 2171 cm⁻¹). The basis set matters as much as the method: CCSD(T)/cc-pVDZ gives 1.144 Å and 2146 cm⁻¹, and the DZ → TZ trend shows that the TZ result is not yet converged; its excellent frequency is partly cancellation. The dipole moment is the clearest warning: Hartree–Fock gets even the sign wrong (−0.24 D), B3LYP and CCSD get the observed C⁻O⁺ polarity. Its derivative, −3.5 D/Å, makes the CO stretch a strong infrared absorber despite the tiny dipole. I trust the CCSD(T)/cc-pVTZ curve most; the next test would be cc-pVQZ (and core correlation) to confirm convergence, and the vibrational levels on this curve (Lab 3) to compare with the measured fundamental rather than ν̃e.

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
# (Instructor version: students write their reflection here.) The lab ran smoothly; the basis-set comparison was the most surprising part.

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
# 1. Q. Sun et al., "Recent developments in the PySCF program package," *J. Chem. Phys.* **153**, 024109 (2020), [doi:10.1063/5.0006074](https://doi.org/10.1063/5.0006074).
# 2. P. Atkins, J. de Paula and J. Keeler, *Atkins' Physical Chemistry*, 11th ed., Oxford University Press (2018): Focus 9 (molecular orbital theory and computational chemistry) and Topic 11C (vibrational spectroscopy of diatomic molecules).
# 3. D. A. McQuarrie and J. D. Simon, *Physical Chemistry: A Molecular Approach*, University Science Books (1997), Ch. 5 (harmonic oscillator), 11 (computational quantum chemistry), 13 (vibrational spectroscopy).
# 4. C. J. Cramer, *Essentials of Computational Chemistry*, 2nd ed., Wiley (2004), Ch. 6–7 (Hartree–Fock, correlated methods).
# 5. C. Møller and M. S. Plesset, "Note on an approximation treatment for many-electron systems," *Phys. Rev.* **46**, 618 (1934), [doi:10.1103/PhysRev.46.618](https://doi.org/10.1103/PhysRev.46.618).
# 6. K. Raghavachari, G. W. Trucks, J. A. Pople and M. Head-Gordon, "A fifth-order perturbation comparison of electron correlation theories," *Chem. Phys. Lett.* **157**, 479 (1989), [doi:10.1016/S0009-2614(89)87395-6](https://doi.org/10.1016/S0009-2614(89)87395-6).
# 7. R. J. Bartlett and M. Musiał, "Coupled-cluster theory in quantum chemistry," *Rev. Mod. Phys.* **79**, 291 (2007), [doi:10.1103/RevModPhys.79.291](https://doi.org/10.1103/RevModPhys.79.291).
# 8. T. H. Dunning Jr., "Gaussian basis sets for use in correlated molecular calculations. I.," *J. Chem. Phys.* **90**, 1007 (1989), [doi:10.1063/1.456153](https://doi.org/10.1063/1.456153).
# 9. G. E. Scuseria, M. D. Miller, F. Jensen and J. Geertsen, "The dipole moment of carbon monoxide," *J. Chem. Phys.* **94**, 6660 (1991), [doi:10.1063/1.460293](https://doi.org/10.1063/1.460293).
# 10. K. P. Huber and G. Herzberg, *Molecular Spectra and Molecular Structure IV. Constants of Diatomic Molecules*, Van Nostrand Reinhold (1979); values via the [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/).

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
lab.submit(name, lab_id='Comp02')
