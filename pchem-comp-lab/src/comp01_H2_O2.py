# LAB
# %% [markdown]
# # Computational Lab 1: From the Hydrogen Atom to O₂
# *CHEM 3376 — Physical Chemistry Laboratory: Data Analysis & Computation*
#
# In this lab you will run quantum chemistry calculations with [PySCF](https://pyscf.org), a free Python package.
# You will start with systems whose **exact** answer is known, so that you can see how close a calculation gets and why.
# Then you will move to a molecule where no exact answer is possible:
#
# 1. the **hydrogen atom** (one electron): the exact energy is −½ hartree, and you will watch the calculation approach it;
# 2. **H₂⁺** (one electron, two nuclei): the simplest chemical bond, also solved exactly;
# 3. **H₂** (two electrons): electron–electron repulsion appears, and with it **electron correlation**;
# 4. **O₂** (sixteen electrons): an exact calculation is impossible, so you will use **density functional theory**, and
#    explain why O₂ is magnetic and why it does not absorb infrared light.
#
# You don't need to know Python already. Each code cell explains what it does, 🐍 **Python tip** boxes introduce one idea
# at a time, and most answers are one line. Stuck? Open the 💡 **Hint** under each question. Cells marked
# *"Just run this cell"* hold longer code that you don't need to read.
#
# <br>**<center>Learning Goals**
#
# |Area|Concept|
# |---|---|
# |Quantum mechanics|Born–Oppenheimer approximation; potential energy curves; the variational principle|
# |Electronic structure|Basis sets; Hartree–Fock; full configuration interaction; electron correlation; DFT|
# |Chemical bonding|The one-electron bond in H₂⁺; bond length, bond energy and force constant; spin states of O₂|
# |Spectroscopy|Harmonic vibrational frequency; why homonuclear diatomics do not absorb IR light|
# |Computation|Why exact calculations are only possible for very small systems|
# |Python|Functions, list comprehensions, `numpy` arrays, plotting, passing a function to a function|

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
ver = '2026Comp01V1'

# %% [markdown]
# ### 🧰 Standard imports
# `numpy` for math on arrays, `matplotlib` for plots, `math` for counting, and four parts of PySCF:
# `gto` (molecules and basis sets), `scf` (Hartree–Fock), `fci` (full configuration interaction) and `dft` (density functional theory).
# The last lines load this lab's **answer checks**: after each question, run `lab.check(...)`. A ✅ means you got it;
# a ❌ tells you what to fix. Fix the cell and run the check again, as often as you like: there is no penalty for trying.
# Every attempt is recorded, and `lab.score()` at the end shows your score.

# %%
# Standard imports for this lab
import numpy as np
import math
%matplotlib inline
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci, dft

# Answer checks for this lab (pchem_check.py and checks_comp01.py are in this folder)
from pchem_check import Checker
from checks_comp01 import CHECKS
lab = Checker('comp01', CHECKS)

# %% [markdown]
# Now run the cell below to enter your name.

# %%
# Enter your name as a string, e.g. name = 'Linus Pauling'
name = 'Instructor'  # SOLUTION

# %% [markdown]
# 🐍 **Python tip — variables and units.** `HARTREE_TO_EV = 27.211386` stores a number under a name. Python doesn't track
# units, so **we put the units in the name or in a comment**. Constants are from
# [CODATA 2018 (NIST)](https://physics.nist.gov/cuu/Constants/).

# %%
# Constants used in this lab
HARTREE_TO_EV = 27.211386        # eV per hartree
BOHR_TO_ANG   = 0.529177         # Å per bohr
HARTREE_TO_J  = 4.3597447e-18    # J per hartree
AMU_TO_KG     = 1.66053907e-27   # kg per atomic mass unit
C_CM_PER_S    = 2.99792458e10    # speed of light, cm s⁻¹
M_H, M_O      = 1.007825, 15.994915   # atomic masses (amu) of ¹H and ¹⁶O

print(f"1 hartree = {HARTREE_TO_EV} eV;  1 bohr = {BOHR_TO_ANG} Å")

# %% [markdown]
# ## 1. Background: Energies for Fixed Nuclei

# %% [markdown]
# ### 1.1 The Born–Oppenheimer approximation
# A proton is 1836 times heavier than an electron, so the electrons adjust almost instantly whenever the nuclei move.
# We can therefore **freeze the nuclei**, solve the Schrödinger equation for the electrons alone, and repeat for many
# nuclear positions. The electronic energy as a function of the nuclear positions is the **potential energy surface**.
# For a diatomic molecule it is a curve, $E(R)$, and the nuclei move on it like a ball rolling in a valley.
# ([McQuarrie & Simon, *Physical Chemistry: A Molecular Approach*, Ch. 9–10](https://uscibooks.aip.org/books/physical-chemistry-a-molecular-approach/);
# [Atkins, *Physical Chemistry*, 11th ed.](https://global.oup.com/academic/product/atkins-physical-chemistry-9780198769866), §9A)
#
# $$ E(R) = E_\text{electronic}(R) + V_{nn}(R) $$
#
# where $V_{nn}$ is the repulsion between the nuclei.
#
# ### 1.2 Atomic units
# Quantum chemistry programs work in **atomic units**: energies in **hartree** ($E_h$) and lengths in **bohr** ($a_0$).
# In these units the hydrogen atom's energy is exactly $-\tfrac12\,E_h$.
#
# | quantity | atomic unit | in everyday units |
# |---|---|---|
# | energy | 1 hartree ($E_h$) | 27.2114 eV = 2625.5 kJ mol⁻¹ |
# | length | 1 bohr ($a_0$) | 0.529177 Å |
#
# In PySCF, **you give positions in Å and it returns energies in hartree.**

# %% [markdown]
# ### <font color=blue> **Question 1.1.** </font>
# Complete the two unit-conversion functions. You'll use them throughout the lab.
# - `hartree_to_eV(E_h)`: hartree → eV
# - `bohr_to_angstrom(R_bohr)`: bohr → Å
#
# Check: the hydrogen atom, $-0.5\ E_h$, should be $-13.606$ eV.
#
# 🐍 **Python tip — functions.** `def name(input):` defines a reusable recipe and `return` hands back the answer.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# Multiply by the constant defined above: `return E_h * HARTREE_TO_EV`. The second function works the same way.
# </details>

# %%
def hartree_to_eV(E_h):
    """Energy: hartree → eV."""
    return E_h * HARTREE_TO_EV  # SOLUTION

def bohr_to_angstrom(R_bohr):
    """Length: bohr → Å."""
    return R_bohr * BOHR_TO_ANG  # SOLUTION

print(f"H atom: -0.5 hartree = {hartree_to_eV(-0.5):.3f} eV")
print(f"2 bohr = {bohr_to_angstrom(2.0):.4f} Å")

# %%
lab.check('1.1')

# %% [markdown]
# ## 2. Your First Calculation: the Hydrogen Atom

# %% [markdown]
# ### 2.1 Orbitals are built from a basis set
# The computer can't store a function exactly. Instead it builds each orbital from a fixed set of simpler functions
# centred on the atoms, the **basis set**, and finds the best combination. A larger basis set gives the orbital more freedom.
# Some common basis sets, from smallest to largest:
#
# | basis set | functions on H | idea |
# |---|---|---|
# | `sto-3g` | 1 | one 1s function (a "minimal" basis) |
# | `6-31g` | 2 | two 1s functions, one tight and one loose |
# | `cc-pvdz`, `cc-pvtz`, `cc-pvqz` | 5, 14, 30 | "correlation-consistent" double, triple, quadruple zeta: more s functions plus p, d, f functions |
#
# For a **one-electron** system there is no electron–electron repulsion, so Hartree–Fock (HF) is the exact Schrödinger
# equation. The **only** error is the basis set.
#
# **Just run the cell below.** It is a complete calculation: build the atom, choose the method, solve.

# %%
# Just run this cell: your first quantum chemistry calculation
mol = gto.M(atom='H 0 0 0',      # one H atom at the origin (x y z in Å)
            basis='sto-3g',      # the basis set
            spin=1,              # number of unpaired electrons (H has one)
            verbose=0)           # don't print PySCF's long log
mf = scf.UHF(mol)                # the method: (unrestricted) Hartree–Fock
E_sto3g = mf.kernel()            # solve; the energy is in hartree

print(f"E(H, STO-3G) = {E_sto3g:.6f} hartree  = {E_sto3g * HARTREE_TO_EV:.3f} eV")
print(f"exact        = -0.500000 hartree  = {-0.5 * HARTREE_TO_EV:.3f} eV")
print(f"error        = {(E_sto3g + 0.5) * HARTREE_TO_EV:.3f} eV")

# %% [markdown]
# ### <font color=blue> **Question 2.1.** </font>
# Turn the worked example into a function, `h_atom_energy(basis)`, that returns the energy (hartree) of an H atom
# for any basis set. The first line is done for you. Then run the cell: it loops over the basis sets and plots the error.
#
# <details><summary>💡 <b>Hint 1</b></summary>
#
# You need the two lines that come after `mol = ...` in the worked example: one that chooses the method, one that solves.
# </details>
# <details><summary>💡 <b>Hint 2</b></summary>
#
# `mf = scf.UHF(mol)` and then `return mf.kernel()`.
# </details>

# %%
def h_atom_energy(basis):
    """Energy (hartree) of a hydrogen atom in the given basis set."""
    mol = gto.M(atom='H 0 0 0', basis=basis, spin=1, verbose=0)
    mf = scf.UHF(mol)  # SOLUTION
    return mf.kernel()  # SOLUTION

basis_sets = ['sto-3g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'cc-pvqz']
E_H = {}                                   # a dictionary: basis set → energy
for b in basis_sets:
    E_H[b] = h_atom_energy(b)
    print(f"{b:8s}  E = {E_H[b]:.6f} hartree   error = {1000 * hartree_to_eV(E_H[b] + 0.5):8.2f} meV")

plt.semilogy(basis_sets, [1000 * hartree_to_eV(E_H[b] + 0.5) for b in basis_sets], 'o-')
plt.ylabel('error above the exact energy (meV)')
plt.title('H atom: approaching the exact answer')
plt.show()

# %%
lab.check('2.1')

# %% [markdown]
# ### <font color=blue> **Question 2.2 Discussion.** </font>
# <font color=green>
#
# 1. Every energy you calculated is **above** the exact −0.5 hartree, never below. This is the **variational principle**.
#    State it in your own words, and explain why a bigger basis set can only lower the energy.
# 2. How large a basis set do you need for an error below 1 meV? Below 1 kJ mol⁻¹ (10 meV)?
# 3. The exact hydrogen 1s orbital is $e^{-r}$, but these basis sets are built from Gaussians, $e^{-\alpha r^2}$. Why would a sum of
#    Gaussians struggle to reproduce the exact orbital near the nucleus, and how does a larger basis help?
# </font>
#
# ---

# %% [markdown] answer
# 1. The energy of any trial wavefunction is greater than or equal to the true ground-state energy, with equality only for the exact wavefunction. A bigger basis contains every combination the smaller one could make, plus more, so the best (lowest) energy it can find is at least as low.
# 2. cc-pVTZ is about 5 meV above exact (below 1 kJ/mol ≈ 10 meV); cc-pVQZ is about 1.5 meV, so close to but not quite below 1 meV. 6-31G is already within 50 meV.
# 3. e^{−r} has a cusp (a sharp point) at the nucleus, but every Gaussian is flat there; a sum of Gaussians can only approximate the cusp with many tight functions. Larger basis sets add more (and tighter) functions, so the error shrinks steadily but never reaches zero with a finite basis.

# %% [markdown]
# ---

# %% [markdown]
# #### Save notebook before check

# %%
lab.check('2.2d')

# %% [markdown]
# ## 3. The Simplest Chemical Bond: H₂⁺

# %% [markdown]
# H₂⁺ has two protons and **one** electron, so, like the H atom, Hartree–Fock is exact and only the basis set limits the answer.
# Its Schrödinger equation was first solved exactly by Burrau in 1927, the first quantum-mechanical treatment of a chemical bond.
# The exact Born–Oppenheimer results ([Bates, Ledsham & Stewart, *Phil. Trans. R. Soc. A* **246**, 215 (1953)](https://doi.org/10.1098/rsta.1953.0014)):
#
# | | exact |
# |---|---|
# | bond length $R_e$ | 1.997 bohr = **1.057 Å** |
# | energy at $R_e$ | **−0.602634** hartree |
# | bond energy $D_e = E(\text{H}) + E(\text{H}^+) - E(R_e)$ | 0.102634 hartree = **2.793 eV** |
#
# The products of breaking the bond are an H atom and a bare proton, H⁺, whose energy is zero (it has no electrons).
#
# To describe an ion you tell PySCF its **charge** and **spin** (the number of unpaired electrons).

# %% [markdown]
# ### <font color=blue> **Question 3.1.** </font>
# Complete `h2plus_energy(R)`: set the `charge` and `spin` for H₂⁺, and return the energy, as in Question 2.1.
# The two H atoms are at $(0, 0, 0)$ and $(0, 0, R)$, with $R$ in Å.
#
# <details><summary>💡 <b>Hint 1</b></summary>
#
# Two protons (+2) and one electron (−1): what is the total charge? How many electrons are unpaired?
# </details>
# <details><summary>💡 <b>Hint 2</b></summary>
#
# `charge = 1`, `spin = 1`, and `return mf.kernel()`.
# </details>

# %%
def h2plus_energy(R, basis='cc-pvtz'):
    """Total energy (hartree) of H2+ with bond length R (Å)."""
    charge = 1  # SOLUTION
    spin = 1  # SOLUTION
    mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0, 0, R)]], basis=basis,
                charge=charge, spin=spin, verbose=0)
    mf = scf.UHF(mol)
    return mf.kernel()  # SOLUTION

print(f"E(H2+, R = 1.057 Å) = {h2plus_energy(1.057):.6f} hartree   (exact at R_e: -0.602634)")

# %%
lab.check('3.1')

# %% [markdown]
# ### <font color=blue> **Question 3.2.** </font>
# Compute the potential energy curve: the energy at every bond length in `R_grid`. Store the results in the array `E_h2p`.
#
# 🐍 **Python tip — list comprehensions.** `[f(R) for R in R_grid]` calls `f` once for each value in `R_grid` and collects the
# results in a list. `np.array(...)` turns the list into an array, so you can do math on all the values at once.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `E_h2p = np.array([h2plus_energy(R) for R in R_grid])`
# </details>

# %%
R_grid = np.linspace(0.5, 4.0, 36)         # bond lengths, Å
E_h2p = np.array([h2plus_energy(R) for R in R_grid])  # SOLUTION

plt.plot(R_grid, E_h2p, 'o-', label='H₂⁺, cc-pVTZ')
plt.axhline(-0.5, color='gray', ls='--', label='H + H⁺ (exact)')
plt.plot(1.057, -0.602634, '*', ms=15, color='k', label='exact minimum')
plt.xlabel('R (Å)'); plt.ylabel('E (hartree)'); plt.ylim(-0.62, -0.2)
plt.title('Potential energy curve of H₂⁺'); plt.legend()
plt.show()

# %%
lab.check('3.2')

# %% [markdown]
# ### 3.3 Bond length, bond energy and force constant
# Near its minimum every smooth curve looks like a parabola,
#
# $$ E(R) \approx E_\text{min} + \tfrac12\,k\,(R - R_e)^2 , $$
#
# where $k$, the curvature, is the bond's **force constant**: a stiff bond has a large $k$.
# The helper below finds $R_e$, $E_\text{min}$ and $k$ by fitting a polynomial to a few closely spaced points near the minimum.
#
# 🐍 **Python tip — passing a function to a function.** `harmonic_fit(h2plus_energy, 1.06)` hands your *function* (no parentheses!)
# to the helper, which calls it as many times as it needs.

# %%
# Just run this cell: fits a polynomial near the minimum of any energy function
def harmonic_fit(energy_function, R_guess, **options):
    """Return R_e (Å), E_min (hartree) and k = E''(R_e) (hartree/Å²) for energy_function(R, **options)."""
    center = R_guess
    for _ in range(2):                                     # fit twice, re-centring on the minimum
        R = center + np.linspace(-0.06, 0.06, 7)           # 7 points within ±0.06 Å
        E = np.array([energy_function(r, **options) for r in R])
        p = np.polyfit(R - center, E, 4)                   # quartic in (R - center)
        roots = np.roots(np.polyder(p))                    # where the slope is zero
        x = roots[np.argmin(np.abs(roots))].real           # the stationary point nearest the centre
        R_e, E_min, k = center + x, np.polyval(p, x), np.polyval(np.polyder(p, 2), x)
        center = R_e
    return R_e, E_min, k

R_e_h2p, E_min_h2p, k_h2p = harmonic_fit(h2plus_energy, 1.06)
print(f"H2+: R_e = {R_e_h2p:.4f} Å,  E_min = {E_min_h2p:.6f} hartree,  k = {k_h2p:.4f} hartree/Å²")

# %% [markdown]
# ### <font color=blue> **Question 3.3.** </font>
# Compute the bond energy of H₂⁺ in eV, `De_h2p`. Use the H atom energy **in the same basis set** (`E_H['cc-pvtz']`), so
# that the basis-set errors partly cancel. Compare with the exact 2.793 eV.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# $D_e = E(\text{H}) + E(\text{H}^+) - E_\text{min}$ with $E(\text{H}^+) = 0$, converted to eV:
# `De_h2p = hartree_to_eV(E_H['cc-pvtz'] - E_min_h2p)`
# </details>

# %%
De_h2p = hartree_to_eV(E_H['cc-pvtz'] - E_min_h2p)  # SOLUTION

print(f"D_e(H2+) = {De_h2p:.3f} eV   (exact 2.793 eV)")
print(f"R_e(H2+) = {R_e_h2p:.4f} Å  (exact 1.057 Å)")

# %%
lab.check('3.3')

# %% [markdown]
# ### 3.4 Where does the bond come from?
# **Just run the cell below.** The left plot splits the energy into the electronic energy and the nuclear repulsion
# $V_{nn} = 1/R$ (atomic units). The right plot shows the electron density along the bond axis at $R_e$, compared with what it
# would be if the electron were simply shared half-and-half between two undisturbed H atoms.

# %%
# Just run this cell: energy components and the electron density of the bond
V_nn = 1 / (R_grid / BOHR_TO_ANG)                       # hartree; 1/R with R in bohr
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(R_grid, E_h2p, 'k-', lw=3, label='total E')
ax[0].plot(R_grid, E_h2p - V_nn, label='electronic energy')
ax[0].plot(R_grid, V_nn, label='nuclear repulsion $V_{nn}$')
ax[0].axhline(0, color='gray', lw=0.8)
ax[0].set_xlabel('R (Å)'); ax[0].set_ylabel('energy (hartree)'); ax[0].set_ylim(-2.2, 1.2); ax[0].legend()
ax[0].set_title('Energy components of H₂⁺')

mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0, 0, R_e_h2p)]], basis='cc-pvtz', charge=1, spin=1, verbose=0)
mf = scf.UHF(mol); mf.kernel()
dm = mf.make_rdm1()[0] + mf.make_rdm1()[1]
z = np.linspace(-1.2, R_e_h2p + 1.2, 500)                 # Å, along the bond axis
pts = np.zeros((len(z), 3)); pts[:, 2] = z / BOHR_TO_ANG  # PySCF wants bohr here
ao = mol.eval_gto('GTOval', pts)
rho_bond = np.einsum('pi,ij,pj->p', ao, dm, ao)
r1, r2 = np.abs(z) / BOHR_TO_ANG, np.abs(z - R_e_h2p) / BOHR_TO_ANG
rho_atoms = 0.5 * (np.exp(-2 * r1) + np.exp(-2 * r2)) / np.pi   # exact 1s densities, half an electron on each
ax[1].plot(z, rho_bond, lw=2.5, label='H₂⁺ (calculated)')
ax[1].plot(z, rho_atoms, '--', lw=2, label='½ H atom + ½ H atom (no bond)')
ax[1].axvline(0, color='gray', lw=0.8); ax[1].axvline(R_e_h2p, color='gray', lw=0.8)
ax[1].set_xlabel('position along the bond (Å)'); ax[1].set_ylabel('electron density (bohr⁻³)')
ax[1].set_ylim(0, 0.27); ax[1].set_title('Electron density along the bond axis'); ax[1].legend(loc='upper center')
plt.tight_layout(); plt.show()

# %% [markdown]
# ### <font color=blue> **Question 3.4 Discussion.** </font>
# <font color=green>
#
# 1. In the density plot, where has the electron gone when the bond forms, compared with two separate half-atoms? Why does
#    putting electron density **between** the nuclei lower the energy?
# 2. In the energy plot, the electronic energy keeps falling as $R$ shrinks, but the total energy has a minimum. What sets
#    the bond length?
# 3. How close are your $R_e$ and $D_e$ to the exact values? What is the only approximation in this calculation?
# </font>
#
# ---

# %% [markdown] answer
# 1. The density piles up between the nuclei (and is depleted just outside them) compared with two undisturbed half-atoms. Density between the nuclei is attracted by both protons at once, which lowers the potential energy; the electron is also spread over a larger region, which lowers its kinetic energy (the original explanation; modern analyses add an orbital contraction near each nucleus).
# 2. The electronic energy falls steadily as the nuclei approach (the electron is attracted by two nuclei that are close together), but the nuclear repulsion 1/R rises steeply at short R. The bond length is where the two balance: the minimum of their sum.
# 3. Reference run: R_e = 1.057 Å and D_e ≈ 2.78 eV (exact 2.793 eV). The only approximation is the finite basis set (and the Born–Oppenheimer approximation, which defines the problem).

# %% [markdown]
# ---

# %% [markdown]
# #### Save notebook before check

# %%
lab.check('3.4d')

# %% [markdown]
# ## 4. Two Electrons: H₂ and Electron Correlation

# %% [markdown]
# ### 4.1 Hartree–Fock is now an approximation
# With two electrons, each electron repels the other. **Hartree–Fock** replaces that instantaneous repulsion by the *average*
# field of the other electron: each electron sees a smeared-out cloud, not a particle it can dodge. The energy it misses is the
# **correlation energy**:
#
# $$ E_\text{corr} = E_\text{exact} - E_\text{HF} \quad (< 0). $$
#
# **Full configuration interaction (FCI)** writes the wavefunction as a combination of *every* possible arrangement
# (determinant) of the electrons in the orbitals. Within a given basis set it is the **exact** answer. Its cost explodes with the
# number of electrons, but for two electrons it takes a fraction of a second.
#
# Exact Born–Oppenheimer values for H₂ ([Kołos & Wolniewicz, *J. Chem. Phys.* **49**, 404 (1968)](https://doi.org/10.1063/1.1669836);
# [Pachucki, *Phys. Rev. A* **82**, 032509 (2010)](https://doi.org/10.1103/PhysRevA.82.032509)) and experiment
# ([Huber & Herzberg, *Constants of Diatomic Molecules* (1979)](https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000)):
#
# | | exact / experiment |
# |---|---|
# | $R_e$ | **0.7414 Å** |
# | $E$ at $R_e$ | **−1.174476** hartree |
# | $D_e = 2E(\text{H}) - E(R_e)$ | 0.174476 hartree = **4.748 eV** |
# | harmonic wavenumber $\tilde\nu_e$ | **4401** cm⁻¹ |

# %% [markdown]
# ### <font color=blue> **Question 4.1.** </font>
# Complete `h2_energy(R, method)`. The Hartree–Fock part is written for you. For `method='fci'`, run full CI on top of the
# Hartree–Fock result with PySCF's `fci.FCI(mf).kernel()`. It returns **two** things, the energy and the wavefunction; keep the energy.
#
# <details><summary>💡 <b>Hint 1</b></summary>
#
# When a function returns two things, `[0]` picks the first: `fci.FCI(mf).kernel()[0]`.
# </details>
# <details><summary>💡 <b>Hint 2</b></summary>
#
# `E_fci = fci.FCI(mf).kernel()[0]`
# </details>

# %%
def h2_energy(R, method='fci', basis='cc-pvtz'):
    """Total energy (hartree) of H2 with bond length R (Å); method is 'hf' or 'fci'."""
    mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0, 0, R)]], basis=basis, verbose=0)
    mf = scf.RHF(mol)            # restricted HF: both electrons in the same spatial orbital
    E_hf = mf.kernel()
    if method == 'hf':
        return E_hf
    E_fci = fci.FCI(mf).kernel()[0]  # SOLUTION
    return E_fci

print(f"HF:  E(R = 0.7414 Å) = {h2_energy(0.7414, 'hf'):.6f} hartree")
print(f"FCI: E(R = 0.7414 Å) = {h2_energy(0.7414, 'fci'):.6f} hartree   (exact -1.174476)")

# %%
lab.check('4.1')

# %% [markdown]
# ### <font color=blue> **Question 4.2.** </font>
# Compute HF and FCI curves on `R_grid2`, as you did for H₂⁺, and store them in `E_hf_curve` and `E_fci_curve`. Then compute the
# correlation energy at every point, `E_corr` (hartree).
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `E_hf_curve = np.array([h2_energy(R, 'hf') for R in R_grid2])`; the FCI curve is the same with `'fci'`.
# `E_corr` is just a subtraction of two arrays.
# </details>

# %%
R_grid2 = np.linspace(0.4, 4.0, 37)        # Å
E_hf_curve = np.array([h2_energy(R, 'hf') for R in R_grid2])  # SOLUTION
E_fci_curve = np.array([h2_energy(R, 'fci') for R in R_grid2])  # SOLUTION
E_corr = E_fci_curve - E_hf_curve  # SOLUTION

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(R_grid2, E_hf_curve, 'o-', label='Hartree–Fock')
ax[0].plot(R_grid2, E_fci_curve, 's-', label='full CI')
ax[0].axhline(2 * E_H['cc-pvtz'], color='gray', ls='--', label='two H atoms')
ax[0].plot(0.7414, -1.174476, '*', ms=15, color='k', label='exact minimum')
ax[0].set_xlabel('R (Å)'); ax[0].set_ylabel('E (hartree)'); ax[0].set_ylim(-1.2, -0.75); ax[0].legend()
ax[0].set_title('H₂, cc-pVTZ')
ax[1].plot(R_grid2, hartree_to_eV(E_corr), 'o-', color='C2')
ax[1].set_xlabel('R (Å)'); ax[1].set_ylabel('correlation energy (eV)')
ax[1].set_title('What Hartree–Fock misses')
plt.tight_layout(); plt.show()

# %%
lab.check('4.2')

# %% [markdown]
# ### <font color=blue> **Question 4.3.** </font>
# The vibrational wavenumber of a diatomic comes from the force constant and the **reduced mass** $\mu = m_1 m_2/(m_1+m_2)$:
#
# $$ \omega = \sqrt{k/\mu}, \qquad \tilde\nu_e = \frac{\omega}{2\pi c}. $$
#
# Complete `harmonic_wavenumber(k, m1, m2)` for $k$ in hartree/Å² and masses in amu. Work in SI units:
# 1. convert $k$ to N m⁻¹ (J m⁻²): multiply by `HARTREE_TO_J` and divide by $(10^{-10})^2 = 10^{-20}$ m² per Å²;
# 2. $\mu$ in kg (`AMU_TO_KG`); 3. $\omega$ in s⁻¹; 4. $\tilde\nu_e$ in cm⁻¹ with $c$ in cm s⁻¹ (`C_CM_PER_S`).
#
# Check: H³⁵Cl has $k$ = 516 N m⁻¹ and $\tilde\nu_e$ = 2991 cm⁻¹.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# ```
# k_SI = k * HARTREE_TO_J / 1e-20
# mu = m1 * m2 / (m1 + m2) * AMU_TO_KG
# omega = np.sqrt(k_SI / mu)
# return omega / (2 * np.pi * C_CM_PER_S)
# ```
# </details>

# %%
def harmonic_wavenumber(k, m1, m2):
    """Harmonic wavenumber (cm⁻¹) from force constant k (hartree/Å²) and atomic masses m1, m2 (amu)."""
    k_SI = k * HARTREE_TO_J / 1e-20    # N/m  # SOLUTION
    mu = m1 * m2 / (m1 + m2) * AMU_TO_KG    # kg  # SOLUTION
    omega = np.sqrt(k_SI / mu)    # s⁻¹  # SOLUTION
    return omega / (2 * np.pi * C_CM_PER_S)  # SOLUTION

print(f"HCl check: {harmonic_wavenumber(1.18424, 1.007825, 34.968853):.0f} cm⁻¹  (expected 2991)")

# %%
lab.check('4.3')

# %% [markdown]
# ### <font color=blue> **Question 4.4.** </font>
# The fits for both methods are done for you. Compute the bond energies `De_hf` and `De_fci` (eV) relative to **two H atoms** in
# the same basis, and the harmonic wavenumbers `nu_hf` and `nu_fci` (cm⁻¹).
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `De_fci = hartree_to_eV(2 * E_H['cc-pvtz'] - E_min_fci)` and `nu_fci = harmonic_wavenumber(k_fci, M_H, M_H)`; the HF lines are the same with `hf`.
# </details>

# %%
R_e_hf, E_min_hf, k_hf = harmonic_fit(h2_energy, 0.74, method='hf')
R_e_fci, E_min_fci, k_fci = harmonic_fit(h2_energy, 0.74, method='fci')

De_hf = hartree_to_eV(2 * E_H['cc-pvtz'] - E_min_hf)  # SOLUTION
De_fci = hartree_to_eV(2 * E_H['cc-pvtz'] - E_min_fci)  # SOLUTION
nu_hf = harmonic_wavenumber(k_hf, M_H, M_H)  # SOLUTION
nu_fci = harmonic_wavenumber(k_fci, M_H, M_H)  # SOLUTION

print(f"{'':10s} {'R_e (Å)':>8s} {'D_e (eV)':>9s} {'ν̃e (cm⁻¹)':>11s}")
print(f"{'HF':10s} {R_e_hf:8.4f} {De_hf:9.3f} {nu_hf:11.0f}")
print(f"{'full CI':10s} {R_e_fci:8.4f} {De_fci:9.3f} {nu_fci:11.0f}")
print(f"{'exact':10s} {0.7414:8.4f} {4.748:9.3f} {4401:11.0f}")

# %%
lab.check('4.4')

# %% [markdown]
# ### <font color=blue> **Question 4.5 Discussion.** </font>
# <font color=green>
#
# 1. The Hartree–Fock curve does not go to the energy of two H atoms when the bond is broken. Restricted HF puts both electrons in
#    the same bonding orbital $\sigma_g = (1s_A + 1s_B)/\sqrt2$. Multiply out $\sigma_g(1)\,\sigma_g(2)$: what fraction of the
#    wavefunction describes H⁺ + H⁻? Why is that wrong at large $R$?
# 2. Why does the correlation energy grow as the bond stretches?
# 3. Which property does correlation change most: $R_e$, $D_e$ or $\tilde\nu_e$? Why does a method that misses correlation
#    give a bond that is too stiff and too weak?
# </font>
#
# ---

# %% [markdown] answer
# 1. σg(1)σg(2) = ½[1sA(1)1sB(2) + 1sB(1)1sA(2)] + ½[1sA(1)1sA(2) + 1sB(1)1sB(2)]: half covalent (H·H) and half ionic (H⁺H⁻, both electrons on one atom). At large R the true state is two neutral H atoms; the ionic half costs about IE − EA ≈ 12.8 eV, so HF goes far too high.
# 2. At R_e the electrons share a compact orbital and the mean-field picture is reasonable. As R grows, the electrons should be on different atoms (left-right correlation), which a single doubly occupied orbital cannot describe, so the error (the correlation energy) grows until it equals the ionic penalty that FCI removes.
# 3. D_e changes most (reference run: HF ≈ 3.6 eV against 4.7 eV for FCI). HF is also too stiff (ν̃e ≈ 4580 against ≈ 4400 cm⁻¹) and slightly too short. Correlation mixes in a little antibonding (σu²) character, which lengthens and softens the bond; without it, the bonding orbital is overpopulated, so the bond is too stiff. The well is too shallow because the HF energy is too high everywhere, and most of all where the electrons should be correlated.

# %% [markdown]
# ---

# %% [markdown]
# #### Save notebook before check

# %%
lab.check('4.5d')

# %% [markdown]
# ## 5. Sixteen Electrons: O₂ with Density Functional Theory

# %% [markdown]
# ### 5.1 Why we can't do full CI for O₂
# The number of determinants in full CI is the number of ways to place the α electrons in the orbitals, times the number of
# ways to place the β electrons:
#
# $$ N_\text{det} = \binom{n_\text{orb}}{n_\alpha}\binom{n_\text{orb}}{n_\beta} $$
#
# H₂ in cc-pVTZ and O₂ in the small cc-pVDZ basis both have **28 orbitals**. H₂ has 1 α and 1 β electron;
# O₂ (a triplet) has 9 α and 7 β.
#
# 🐍 **Python tip:** `math.comb(n, k)` counts the ways to choose `k` things from `n`: `math.comb(28, 1)` is 28.

# %% [markdown]
# ### <font color=blue> **Question 5.1.** </font>
# Compute `n_det_H2` and `n_det_O2`. The cell then estimates the memory needed just to *store* one number per determinant.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `n_det_O2 = math.comb(28, 9) * math.comb(28, 7)`
# </details>

# %%
n_det_H2 = math.comb(28, 1) * math.comb(28, 1)  # SOLUTION
n_det_O2 = math.comb(28, 9) * math.comb(28, 7)  # SOLUTION

print(f"H2: {n_det_H2:,} determinants")
print(f"O2: {n_det_O2:,} determinants  →  {n_det_O2 * 8 / 1e12:.0f} TB just to store the wavefunction")

# %%
lab.check('5.1')

# %% [markdown]
# ### 5.2 Density functional theory
# **Density functional theory (DFT)** avoids the wavefunction altogether. Hohenberg and Kohn proved that the ground-state energy is
# determined by the electron **density** $\rho(\mathbf r)$, a function of just three coordinates however many electrons there are.
# The catch: the exact functional is unknown, so we use an approximate **exchange–correlation functional**. Here we use
# **B3LYP**, the most widely used one, with the **def2-SVP** basis set. A DFT calculation on O₂ takes about a second.
#
# O₂ has 16 electrons. Its two highest electrons go into a pair of degenerate π* orbitals, so it can be a **triplet**
# (one electron in each, spins parallel, `spin=2`) or a **closed-shell singlet** (both in one, paired, `spin=0`).
# Unrestricted DFT (`dft.UKS`) lets the α and β electrons have different orbitals, which an open-shell molecule needs.
# Restricted DFT (`dft.RKS`) pairs them, as for a closed shell.

# %% [markdown]
# ### <font color=blue> **Question 5.2.** </font>
# Complete `o2_energy(R, spin)`: choose `dft.UKS(mol)` when `spin > 0` and `dft.RKS(mol)` otherwise, and return the energy (hartree).
#
# 🐍 **Python tip — choosing in one line.** `a if condition else b` gives `a` when the condition is true, otherwise `b`.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)` and `return mf.kernel()`.
# </details>

# %%
def o2_energy(R, spin=2, xc='b3lyp', basis='def2-svp'):
    """B3LYP total energy (hartree) of O2 with bond length R (Å) and `spin` unpaired electrons."""
    mol = gto.M(atom=[['O', (0, 0, 0)], ['O', (0, 0, R)]], basis=basis, spin=spin, verbose=0)
    mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)  # SOLUTION
    mf.xc = xc
    return mf.kernel()  # SOLUTION

print(f"E(O2, triplet, R = 1.21 Å) = {o2_energy(1.21, spin=2):.6f} hartree")

# %%
lab.check('5.2')

# %% [markdown]
# ### <font color=blue> **Question 5.3.** </font>
# Compute the triplet and singlet curves on `R_grid3` (`E_trip` and `E_sing`, hartree), as for H₂ (about 30 s).
#
# <details><summary>💡 <b>Hint</b></summary>
#
# `E_trip = np.array([o2_energy(R, spin=2) for R in R_grid3])`, and the same with `spin=0` for the singlet.
# </details>

# %%
R_grid3 = np.linspace(1.0, 2.0, 21)        # Å
E_trip = np.array([o2_energy(R, spin=2) for R in R_grid3])  # SOLUTION
E_sing = np.array([o2_energy(R, spin=0) for R in R_grid3])  # SOLUTION

E_ref = E_trip.min()
plt.plot(R_grid3, hartree_to_eV(E_trip - E_ref), 'o-', label='triplet (spin = 2)')
plt.plot(R_grid3, hartree_to_eV(E_sing - E_ref), 's-', label='closed-shell singlet (spin = 0)')
plt.xlabel('R (Å)'); plt.ylabel('E − E_min (eV)'); plt.ylim(-0.2, 6); plt.legend()
plt.title('O₂, B3LYP/def2-SVP'); plt.show()
i = np.argmin(E_trip)
print(f"singlet − triplet at R = {R_grid3[i]:.2f} Å: {hartree_to_eV(E_sing[i] - E_trip[i]):.2f} eV")

# %%
lab.check('5.3')

# %% [markdown]
# ### <font color=blue> **Question 5.4.** </font>
# Reuse your tools. The fit for the triplet and the O atom energy are done for you. Compute the harmonic wavenumber `nu_O2` (cm⁻¹) and
# the bond energy `De_O2` (eV) relative to two O atoms. Experiment: $R_e$ = 1.2075 Å, $\tilde\nu_e$ = 1580 cm⁻¹, $D_e$ = 5.21 eV.
#
# <details><summary>💡 <b>Hint</b></summary>
#
# Exactly as in Question 4.4, with `M_O` and `2 * E_O`.
# </details>

# %%
R_e_O2, E_min_O2, k_O2 = harmonic_fit(o2_energy, 1.21, spin=2)
atom = gto.M(atom='O 0 0 0', basis='def2-svp', spin=2, verbose=0)   # an O atom is a triplet too
mf_O = dft.UKS(atom); mf_O.xc = 'b3lyp'
E_O = mf_O.kernel()

nu_O2 = harmonic_wavenumber(k_O2, M_O, M_O)  # SOLUTION
De_O2 = hartree_to_eV(2 * E_O - E_min_O2)  # SOLUTION

print(f"O2: R_e = {R_e_O2:.4f} Å (expt 1.2075),  ν̃e = {nu_O2:.0f} cm⁻¹ (expt 1580),  D_e = {De_O2:.2f} eV (expt 5.21)")

# %%
lab.check('5.4')

# %% [markdown]
# ### 5.5 Does O₂ absorb infrared light?
# A vibration absorbs infrared light only if it makes the **dipole moment** oscillate: the intensity is proportional to
# $(\partial\mu/\partial R)^2$. **Just run the cell below** to compute the dipole of O₂ at three bond lengths.

# %%
# Just run this cell: dipole moment of O2 at three bond lengths
for R in (1.1, 1.2, 1.3):
    mol = gto.M(atom=[['O', (0, 0, 0)], ['O', (0, 0, R)]], basis='def2-svp', spin=2, verbose=0)
    mf = dft.UKS(mol); mf.xc = 'b3lyp'; mf.kernel()
    print(f"R = {R:.1f} Å:  dipole (x, y, z) = {np.round(mf.dip_moment(unit='Debye', verbose=0), 6)} D")

# %% [markdown]
# ### <font color=blue> **Question 5.5 Discussion.** </font>
# <font color=green>
#
# 1. Which spin state of O₂ is the ground state, and by how much? Draw (on paper) the molecular-orbital diagram, show where the last two
#    electrons go, and use it to explain why liquid oxygen sticks to a magnet.
# 2. Explain from symmetry why the dipole of O₂ (and of H₂ and N₂) is zero at **every** bond length. O₂ and N₂ are 99 % of the
#    atmosphere: what does this mean for how the atmosphere absorbs the infrared light emitted by the Earth?
# 3. For H₂ you could check your answer against an exact one; for O₂ you can't. How does B3LYP do against experiment? Why is it
#    harder to know how much to trust a DFT result than a full CI result?
# </font>
#
# ---

# %% [markdown] answer
# 1. The triplet, about 1.7 eV below the closed-shell singlet in this calculation. MO filling (σ2s)²(σ*2s)²(σ2p)²(π2p)⁴(π*2p)²: the last two electrons occupy the two degenerate π* orbitals with parallel spins (Hund's rule), giving bond order 2 and two unpaired electrons, hence a net magnetic moment (paramagnetism). (The real lowest singlet, a¹Δg at 0.98 eV, is open-shell; the closed-shell gap overestimates it.)
# 2. A homonuclear diatomic has a centre of inversion at every bond length. Inversion turns the dipole vector μ into −μ, so μ = 0 always and dμ/dR = 0: the stretch is IR-inactive. N₂ and O₂ are transparent to the Earth's thermal IR; the greenhouse effect comes from minor gases whose vibrations change the dipole (H₂O, CO₂, CH₄, N₂O, O₃).
# 3. B3LYP/def2-SVP: R_e ≈ 1.200 Å (expt 1.2075), ν̃e ≈ 1700 cm⁻¹ (1580), D_e ≈ 5.5 eV (5.21): good but not exact. Full CI is exact in its basis and can be converged by enlarging the basis (as we saw for H, H₂⁺ and H₂); a DFT functional is approximate in a way that cannot be systematically improved, so its reliability has to be judged by benchmarking against experiment or higher-level theory.

# %% [markdown]
# ---

# %% [markdown]
# #### Save notebook before check

# %%
lab.check('5.5d')

# %% [markdown]
# ### 📋 Summary of your results
# **Just run the cell below.** It collects your results in one table; you will refer to it in the next lab.

# %%
# Just run this cell: summary table
summary = Table().with_columns(
    'System', ['H atom', 'H₂⁺', 'H₂ (HF)', 'H₂ (full CI)', 'O₂ (B3LYP)'],
    'Method / basis', ['HF / cc-pVQZ', 'HF / cc-pVTZ', 'HF / cc-pVTZ', 'FCI / cc-pVTZ', 'B3LYP / def2-SVP'],
    'R_e (Å)', ['—', round(R_e_h2p, 4), round(R_e_hf, 4), round(R_e_fci, 4), round(R_e_O2, 4)],
    'D_e (eV)', ['—', round(De_h2p, 3), round(De_hf, 3), round(De_fci, 3), round(De_O2, 2)],
    'ν̃e (cm⁻¹)', ['—', '—', round(nu_hf), round(nu_fci), round(nu_O2)],
    'E error vs exact (meV)', [round(1000 * hartree_to_eV(E_H['cc-pvqz'] + 0.5), 1), round(1000 * hartree_to_eV(E_min_h2p + 0.602634), 1),
                               round(1000 * hartree_to_eV(E_min_hf + 1.174476), 0), round(1000 * hartree_to_eV(E_min_fci + 1.174476), 1), 'no exact answer'],
    'Reference (exact or expt)', ['E = −0.5 Eh', '1.057 Å, 2.793 eV', '0.7414 Å, 4.748 eV, 4401', '0.7414 Å, 4.748 eV, 4401', '1.2075 Å, 5.21 eV, 1580'])
summary.show()

# %% [markdown]
# ### 📝 Quick Check — From H to O₂
# Test what you've learned. This is just for practice — answer as many times as you like, it isn't recorded.

# %%
with open("questionsComp01.json", "r") as f:
    display_quiz(json.load(f), shuffle_answers=True)

# %% [markdown]
# ## 6. Reflection

# %% [markdown]
# ### <font color=blue> **Question 6.** </font>
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
# (Instructor version: students write their reflection here.) The lab ran smoothly; the hints for Questions 3.1 and 4.1 were the most useful.

# %% [markdown]
# ---

# %% [markdown]
# #### Save notebook before check

# %%
lab.check('6d')

# %% [markdown]
# ## 📚 References
# 1. Q. Sun et al., "Recent developments in the PySCF program package," *J. Chem. Phys.* **153**, 024109 (2020), [doi:10.1063/5.0006074](https://doi.org/10.1063/5.0006074).
# 2. D. A. McQuarrie and J. D. Simon, *Physical Chemistry: A Molecular Approach*, University Science Books (1997), Ch. 9–11 (H₂⁺, H₂, molecular orbitals, computational chemistry).
# 3. P. Atkins and J. de Paula, *Atkins' Physical Chemistry*, 11th ed., Oxford University Press (2018), Focus 9 (molecular structure).
# 4. C. J. Cramer, *Essentials of Computational Chemistry*, 2nd ed., Wiley (2004), Ch. 4–8.
# 5. Ø. Burrau, "Berechnung des Energiewertes des Wasserstoffmolekel-Ions (H₂⁺) im Normalzustand," *Kgl. Danske Vid. Selsk. Math.-fys. Medd.* **7**, 1 (1927).
# 6. D. R. Bates, K. Ledsham and A. L. Stewart, "Wave functions of the hydrogen molecular ion," *Phil. Trans. R. Soc. A* **246**, 215–240 (1953), [doi:10.1098/rsta.1953.0014](https://doi.org/10.1098/rsta.1953.0014).
# 7. W. Kołos and L. Wolniewicz, "Improved theoretical ground-state energy of the hydrogen molecule," *J. Chem. Phys.* **49**, 404–410 (1968), [doi:10.1063/1.1669836](https://doi.org/10.1063/1.1669836).
# 8. K. Pachucki, "Born–Oppenheimer potential for H₂," *Phys. Rev. A* **82**, 032509 (2010), [doi:10.1103/PhysRevA.82.032509](https://doi.org/10.1103/PhysRevA.82.032509).
# 9. K. P. Huber and G. Herzberg, *Molecular Spectra and Molecular Structure IV. Constants of Diatomic Molecules*, Van Nostrand Reinhold (1979); values via the [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/).
# 10. P. Hohenberg and W. Kohn, "Inhomogeneous electron gas," *Phys. Rev.* **136**, B864 (1964), [doi:10.1103/PhysRev.136.B864](https://doi.org/10.1103/PhysRev.136.B864).
# 11. A. D. Becke, "Density-functional thermochemistry. III. The role of exact exchange," *J. Chem. Phys.* **98**, 5648 (1993), [doi:10.1063/1.464913](https://doi.org/10.1063/1.464913).
# 12. E. Tiesinga et al., "CODATA Recommended Values of the Fundamental Physical Constants: 2018," *Rev. Mod. Phys.* **93**, 025010 (2021).

# %% [markdown]
# ## 🏁 Final Step — Check, Then Submit to Canvas
#
# Follow these steps **in order**:
#
# 1. ✅ **Run the cell below** to see your score: one row per question, with the number of attempts. Any question still marked ❌ or *not checked*: go back, fix it, and run its check again.
# 2. 💾 **Save** the notebook (Ctrl/Cmd + S).
# 3. 📤 **Make a PDF and upload it to Canvas.** Export an HTML copy ([instructions](https://laserchemist.github.io/data/submission-instructions/export-html)), open it in your browser, and use **Print → Save as PDF**. Check that your plots, tables and written answers all appear in the PDF, then upload it to this lab's Canvas assignment.

# %%
# Your score for this lab
lab.score()
