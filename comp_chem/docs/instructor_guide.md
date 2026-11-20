# Instructor book
### Computational chemistry (CHEM 3376): from electrons to infrared spectra

This book covers all seven computational labs in `comp_chem/`. The **instructor notebooks** in `comp_chem/instructor/<lab>/`
are the other half of it: each holds the full solution, a sample answer for every written question (abstract, discussions,
conclusions), instructor notes, the plain answer key and the outputs of a complete reference run. Open the instructor
notebook next to this book when you prepare a session.

Contents: 1. Purpose and learning goals · 2. Schedule · 3. Setup and distribution · 4. The answer checks · 5. Lab-by-lab
notes (reference values, teaching points, every check and the mistakes it recognises) · 6. Assessment · 7. Troubleshooting ·
8. Maintaining the material

## 1. Purpose and learning goals

The labs take physical chemistry juniors and seniors from the hydrogen atom to a dynamical simulation of an infrared spectrum,
with the physics in front and the code kept short. By the end, students should be able to:

1. explain the variational principle and the Born–Oppenheimer potential energy surface, and compute surfaces with
   Hartree–Fock, full CI, MP2, coupled cluster and DFT, including spin states;
2. explain electron correlation and basis-set convergence, and why wavefunction methods can be improved systematically
   (and why agreement with experiment is not the same as convergence);
3. obtain harmonic frequencies from curvature, and normal modes from the mass-weighted Hessian;
4. solve the 1-D vibrational Schrödinger equation on a grid, and relate anharmonicity, zero-point energy, overtones and isotope
   shifts to the shape of the curve;
5. predict IR activity from symmetry and from dμ/dQ, and connect the CO₂ bend to the greenhouse effect;
6. build and *test* a machine-learned potential energy surface (Gaussian process) with symmetry-adapted coordinates;
7. integrate Newton's equations (velocity Verlet), choose a time step, and compute a spectrum from the Fourier transform of
   the dipole moment;
8. communicate a computational result as an abstract, conclusions and figures.

The ML/MD part follows research practice in a form undergraduates can complete: symmetry built into the inputs, sampling by
normal-mode displacement, held-out test sets, harmonic frequencies checked against the reference, a physically motivated dipole
model, and classical (not ring-polymer) dynamics.

**The course format.** Every lab has the course initialization cell, an **Abstract** cell at the top (written last), short
scaffolded code (`x = ...` blanks, usually one to three per question) with a 💡 hint under each question, answer checks after
each question (`lab.check('2.1')`), discussion questions answered in markdown, a practice quiz (jupyterquiz), **Conclusions**,
a **Reflection**, `lab.score()` and the course **Submit** button. Students then export a PDF to Canvas.

## 2. Schedule

| | content | time | compute (2 cores) | notes |
|---|---|---|---|---|
| **Session 1** | Introduction (15 min): PES, DFT vs wavefunction methods, Jupyter basics | | | running the first H-atom example together helps |
| | Lab 1 `comp01_H2_O2`: H atom → Gaussians → variational principle → H₂⁺ → H₂ → O₂ | 100 min | 1 min | |
| | Lab 2 `comp02_CO_ab_initio`: CO ab initio and basis sets | 45 min | 2 min | saves the CO curve for Lab 3 |
| | Lab 3 `comp03_CO_vibrations`: vibrational levels | 40 min | seconds | the most mathematical lab; can be finished at home |
| **Session 2** | Recap (10 min): normal modes, IR selection rule | | | |
| | Lab 4 `comp04_CO2_normal_modes`: CO₂ harmonic analysis and IR | 50 min | 25 s | saves modes for Labs 5–6 |
| | Lab 5 `comp05_CO2_ML_surface`: ML surface | 60 min | 1.5 min | saves the surface for Lab 6 |
| | Lab 6 `comp06_CO2_MD_IR`: MD and the IR spectrum | 60 min | 3 min | |
| | Project briefing (10 min): assign molecules, show an example abstract | | | |
| **Week** | Lab 7 `comp07_project`: project, abstract and figures | 4–6 h of student time | 5–10 min | an office hour midway helps |

Labs 2 → 3 and 4 → 5 → 6 pass results through `comp_chem/results/`, but each lab loads the reference copy from `data/` if the
student's file is missing (it prints which it used), so nobody is blocked. Lab 6 refits the reference surface (2 s) if Lab 5's
is missing.

**Assign project molecules** so that the class covers several. Good pairings for discussion: N₂O vs CS₂ (both linear, with and
without an inversion centre), H₂O vs SO₂ (both bent; light vs heavy atoms). HCN and H₂O need Δt = 0.25 fs (set automatically).

## 3. Setup and distribution

### Packages
`pip install -r comp_chem/requirements.txt` (pyscf, numpy, scipy, matplotlib, scikit-learn, py3Dmol). The course's
`shared-public/` must provide `init.py`/`init_access.py` (`display_quiz`, `Table`, `json`, styling) and `lab_submit.py`; gofer
is not needed by these labs (the init files import it only if it is installed). PySCF has Linux and macOS wheels; it does not
support native Windows, so students need the hub (or WSL).

### What students open
`compute.html` in the repository root (linked from the course's `index.html`) has one nbgitpuller button per lab:
`https://temple.2i2c.cloud/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Flaserchemist%2Fpchem&urlpath=lab%2Ftree%2Fpchem%2Fcomp_chem%2F<lab>%2F<lab>.ipynb&branch=main`.
The first click pulls the repository into `~/pchem`; later clicks update it without overwriting the student's work
(nbgitpuller merges, keeping local changes).

### ⚠️ Keep solutions out of the students' copy
nbgitpuller copies the **whole repository** to every student, so anything in a public repository is visible to students.
`comp_chem/instructor/` (solutions, sample answers, plain answer keys) and `comp_chem/src/` (the sources, which contain the
solutions, and the plain keys) should therefore live in a separate **private** repository for staff, with the public
repository holding only the lab folders, `labkit/`, `data/`, `docs/student_handout.md` and `compute.html`. The build tool
writes both sets, so the simplest workflow is: keep the full tree in the private repository, and copy the student folders
to the public one after each build.

### Server resources
PySCF uses OpenMP and by default takes every core. On a shared hub, set `OMP_NUM_THREADS=2` in the spawner environment; the
compute times in §2 are for 2 cores. Memory stays well under 1 GB per student; the largest job is the optional
CCSD(T)/cc-pVQZ for CO (~0.5 GB).

### Before the first session
Open each lab once on the hub with a test account and run it top to bottom with the instructor notebook beside it (or run
`python tools/build_notebooks.py --execute` on the hub, about 10 minutes on 8 cores). Check that the quiz and the 3-D viewer
appear (the viewer needs JavaScript; without py3Dmol a matplotlib fallback appears), and click Submit once to confirm that
the record reaches `shared-public/submissions.json`.

## 4. The answer checks

The checks are the course's own, not gofer: `pchem_check.py` (one master copy, `src/_shared/pchem_check.py`, copied into
every lab folder by the build) reads the lab's answer key `answers_compNN.json`.

- **What students see.** `lab.check('4.1')` runs that question's tests on the notebook's variables and stops at the first
  failure. A pass shows ✅ with one sentence on why the result matters. A miss shows ❌ with feedback for *that* mistake: first
  the key's list of typical wrong answers for the test, then the lab's factor diagnoses (a missing hartree → eV conversion, a
  factor 1000, 2π, ...) and a wrong sign, then the test's hint. A leftover `...`, a missing `return` and an array index used
  as a value get their own messages. Students can retry as often as they like.
- **Written answers** (abstract, discussions, conclusions, reflection) are "open" checks: they pass when the markdown cell under
  the question holds enough of the student's own writing (60 characters for discussions, 300 for conclusions, 400 for the
  abstract; 600 for the Lab 7 abstract), read from the *saved* notebook. They are checked for presence, not graded.
- **The record.** Every attempt is logged with a time stamp in `.compNN_check_log.json` next to the notebook, so it survives
  a kernel restart; `lab.score()` shows each question's status and attempts. The log is a plain file a determined student
  could edit: use it for feedback, not as the grade.
- **Submission.** `lab.submit(name, lab_id='Comp04')` calls `submit_lab` in `shared-public/lab_submit.py` with the score and
  the saved text of every written answer (`open_answers`). Records go to `shared-public/submissions.json`
  (or `~/submissions_local.json`). Students can resubmit; the latest counts.
- **Encoding.** The students' copy of each key is compressed and scrambled (format `pchem-key-1`) so it can't be read by
  opening or searching the file. This is not security: the checks run in the student's own kernel, so a determined student
  can decode the key, exactly as with gofer's test files. For grading, re-run submitted notebooks.

### The answer-key format
Each question has `tests`; each test is a Python expression in the notebook's variables, either
`{"value": "De_h2p", "answer": 2.787, "tol": 0.05, "hint": "...", "wrong": [...]}` or `{"assert": "len(E) == len(R)", ...}`.
A `wrong` entry is `{"answer": 16.39, "feedback": "You left out E(H)."}` or `{"when": "value > 100", "feedback": ...}`, where
`value` is the test's value and `answer` the key's answer. Optional fields: `needs` (names that must exist), `setup` (Python
run first, on a copy of the namespace, e.g. to build a reference from the student's own data), `show_value: false`, and `ok`
(the message on success). Labs whose numbers depend on each student's data or molecule (Lab 7) use asserts on physics
(mode count, energy conservation, a finite-difference check of the gradient) rather than fixed numbers. The full format is in
the docstring of `pchem_check.py`.

**Editing a key:** edit the plain `src/<lab>/answers_compNN.json`, rebuild (`python tools/build_notebooks.py --only compNN`),
and run the lab once. The reference values in the keys came from complete runs of the instructor notebooks; every key was
also tested against simulated typical mistakes (§5), each of which gets its targeted message. By hand:
`python pchem_check.py encode KEY.json [OUT]` and `python pchem_check.py decode KEY.json [OUT]`.

## 5. Lab-by-lab notes

Reference values are from the executed instructor notebooks (PySCF 2.14, scikit-learn 1.9; B3LYP/def2-SVP unless stated).
Students' values agree closely; the ML and MD numbers use fixed random seeds, so they are reproducible unless a student
changes the order of the random draws.

### Lab 1 (`comp01_H2_O2`): from the hydrogen atom to O₂
25 checks (7 written). Abstract ≥ 400 characters, Conclusions `7c`, Reflection `8d`.

The order builds from systems with exact answers to one without:

- **H atom, exact.** Radial functions R_nl and radial distribution functions for n = 1–3 (Atkins Table 8A.1); students
  code R_1s, P(r) and find r_mp = 1 bohr.
- **Gaussians.** Students code a normalized Gaussian and a contracted sum; STO-nG fits (ζ = 1; Hehre–Stewart–Pople 1969,
  Stewart 1970) give R(0) = 0.95, 1.38, 1.61, 1.88 for n = 1, 2, 3, 6 against the exact 2: the missing cusp.
- **Variational principle** (boxed statement, proof sketch, Hylleraas history box). One Gaussian: α = 8/9π = 0.283,
  E = −4/3π = −0.42441 Eh (STO-1G, a shape fit, is slightly higher at −0.42422: a good discussion point). Helium with
  e^{−ζ(r1+r2)}: ζ = 27/16, E = −2.84766 Eh (exact −2.903724; HF limit −2.861680). STO-nG energies with PySCF custom
  basis: −0.42422, −0.48116, −0.49491, −0.49983 Eh.
- **Library basis sets.** Errors above −0.5 Eh: STO-3G 909 meV, 6-31G 48, cc-pVDZ 20, cc-pVTZ 5.2, cc-pVQZ 1.5 meV. PySCF's
  STO-3G hydrogen uses ζ = 1.24 (molecular), so it is worse for the free atom than the ζ = 1 STO-3G (139 meV).
- **H₂⁺ LCAO.** Students code S(R) (S(2) = 0.5865) and ψ± along the axis. The plot compares LCAO with the exact-in-basis
  cc-pVTZ orbitals (diagonalizing the one-electron Hamiltonian) at R = 2 and 5 bohr: σg −0.6022 vs LCAO −0.5538 Eh,
  σu* −0.1671 vs −0.1609 at 2 bohr; −0.5221/−0.5192 and −0.4770/−0.4766 at 5 bohr. Minimal LCAO curve: R_e = 1.32 Å,
  D_e = 1.76 eV. Optimizing ζ in the LCAO at R = 2 bohr gives ζ ≈ 1.24 and D_e = 2.35 eV.
- **H₂⁺** (one electron, exact solution known): cc-pVTZ gives R_e = 1.0572 Å (exact 1.057), E = −0.602245 Eh
  (exact −0.602634), D_e = 2.787 eV (exact 2.793). The density plot shows charge built up between the nuclei; the
  energy-component plot shows the bond length as a balance between electronic attraction and 1/R.
- **H₂** (two electrons): full CI is exact in the basis and takes a fraction of a second. cc-pVTZ: HF 0.7344 Å, 3.63 eV,
  4587 cm⁻¹; FCI 0.7426 Å, 4.70 eV, 4409 cm⁻¹; exact 0.7414 Å, 4.748 eV, 4401 cm⁻¹. RHF dissociates far too high
  (half ionic); the correlation energy grows from −1.0 eV at R_e to −5.7 eV at 4 Å.
- **O₂**: 784 determinants for H₂ against 8.2 × 10¹² for O₂ in the same number (28) of orbitals, which motivates DFT.
  B3LYP/def2-SVP: triplet ground state, closed-shell singlet 1.71 eV higher; R_e 1.200 Å (expt 1.2075), 1697 cm⁻¹
  (expt 1580), D_e 5.53 eV (expt 5.21); dipole exactly zero.
- *Teaching points:* the real lowest singlet (a¹Δg, 0.98 eV) is open-shell, so the closed-shell gap overestimates it. Use
  `dft.RKS` for the closed-shell singlet; `UKS` with spin 0 breaks symmetry and gives a misleading 0.45 eV.
- *Common errors:* charge/spin for H₂⁺ (charge 1, spin 1); forgetting `[0]` after `fci.FCI(mf).kernel()`; the 10⁻²⁰
  in the force-constant conversion.
- *Setup:* the lab needs the course's `shared-public/` (`init.py` / `init_access.py`, which provide `display_quiz`, `Table`,
  `json`). PySCF must be installed on the hub image. gofer is no longer used by this lab.

**Checks** and the typical mistakes they recognise

| check | what is tested (reference) | typical mistakes recognised |
|---|---|---|
| `abstract` | Abstract: written answer, ≥ 400 characters in the saved notebook | — |
| `1.1` | **Unit conversions.** `hartree_to_eV(1.0)` = 27.21 ± 0.001; `hartree_to_eV(-0.5)` = -13.61 ± 0.001; `bohr_to_angstrom(2.0)` = 1.058 ± 0.0001 | You divided by `HARTREE_TO_EV`; `hartree_to_eV` hands back its input unchanged; You divided by `BOHR_TO_ANG`; `bohr_to_angstrom` hands back its input unchanged |
| `2.1` | **Hydrogen 1s radial function.** `R_1s(0.0)` = 2 ± 1e-06; `R_1s(1.0)` = 0.7358 ± 1e-06; `np.allclose(P_1s, r**2 * 4 * np.exp(-2 * r))`; `r_mp` = 1 ± 0.011 | `R_1s(0)` is 1; That is the whole 1s orbital, ψ = e−r/√π; Your function grows with r; You multiplied by r, not r²; R(r) needs to be squared (a probability is \|ψ\|²); That is R² alone; The extra 4π is already included in R(r)'s normalization; `r_mp` is a whole number; r = 0 is where R(r) (and ψ) is largest, but P(r) is zero there; That is the height of the peak (the largest value of P); A peak at 2 bohr means P(r) used R instead of R² |
| `2.2` | **Gaussians and contracted Gaussians.** `gaussian_R(0.0, 1.0)` = 2.526 ± 1e-05; `gaussian_R(2.0, 0.5)` = 0.2033 ± 1e-06; `sto_ng_R(0.0, [1.0], [1.0])` = 2.526 ± 1e-05; `sto_ng_R(0.0, *STO[3])` = 1.613 ± 0.0001 | Without the normalization constant; Your exponent is −α r; The exponent is missing its minus sign; That is only the last Gaussian; The Gaussians were added without their coefficients |
| `3.1` | **Variational energy of one Gaussian.** `E_gauss(1.0)` = -0.09577 ± 1e-06; `alpha_best` = 0.2829 ± 0.003; `E_gauss_best` = -0.4244 ± 1e-05 | The square root is missing from the potential energy; That is the kinetic energy alone; That is the potential energy alone; `alpha_best` is a whole number; That is the end of the grid, where the energy is highest; That is the lowest energy, not the α where it happens; That is α, not the energy |
| `3.2` | **Helium: Hylleraas's problem.** `E_He(2.0)` = -2.75 ± 1e-06; `zeta_best` = 1.688 ± 0.002; `E_He_best` = -2.848 ± 1e-05 | That is ζ² − 4ζ; Sign; `zeta_best` is a whole number; That is the end of the grid, where the energy is highest; That is the lowest energy, not the ζ where it happens.; That is ζ, not the energy |
| `3.3` | **H atom in several basis sets.** `h_atom_energy('sto-3g')` = -0.4666 ± 1e-05; `all(b in E_H for b in ['sto-3g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'cc-pvqz'])`; `E_H['cc-pvqz']` = -0.4999 ± 1e-05 | — |
| `3.4d` | Discussion: the variational principle: written answer, ≥ 60 characters in the saved notebook | — |
| `4.1` | **H₂⁺ LCAO orbitals.** `overlap_S(0.0)` = 1 ± 1e-09; `overlap_S(2.0)` = 0.5865 ± 1e-05; `psi_lcao(1.0, 2.0, -1)` = 0 ± 1e-09; `psi_lcao(1.0, 2.0, +1)` = 0.233 ± 1e-05; `psi_lcao(0.0, 2.0, -1)` = 0.5364 ± 1e-05 | The `R**2 / 3` term is missing; R² should be divided by 3; The factor `np.exp(-R)` is missing; σu* came out the same as σg; You divided by √2 only; The orbital isn't normalized; You divided by 2(1 ± S); For σu* the normalization uses 1 − S |
| `4.2` | **H₂⁺ energy function.** `h2plus_energy(1.057)` = -0.6022 ± 2e-05 | The energy is far too low for one electron; The energy is too high for H₂⁺; Close, but not the cc-pVTZ value |
| `4.3` | **H₂⁺ potential energy curve.** `len(E_h2p) == len(R_grid)`; `R_grid[np.argmin(E_h2p)]` = 1.06 ± 0.1 | `E_h2p` is a single number; Every energy is the same |
| `4.4` | **H₂⁺ bond energy.** `De_h2p` = 2.787 ± 0.05 | About 16 eV; That is E(H) on its own, in eV |
| `4.5d` | Discussion: where the bond comes from: written answer, ≥ 60 characters in the saved notebook | — |
| `5.1` | **H₂ energy: Hartree–Fock and full CI.** `h2_energy(0.7414, 'hf')` = -1.133 ± 2e-05; `np.ndim(h2_energy(0.7414, 'fci')) == 0`; `h2_energy(0.7414, 'fci')` = -1.172 ± 2e-05 | That is the Hartree–Fock energy |
| `5.2` | **H₂ curves and correlation energy.** `len(E_hf_curve) == len(R_grid2) and len(E_fci_curve) == len(R_grid2)`; `np.all(np.asarray(E_fci_curve) <= np.asarray(E_hf_curve) + 1e-8)`; `np.allclose(E_corr, np.asarray(E_fci_curve) - np.asarray(E_hf_curve))` | Right size, wrong sign |
| `5.3` | **Harmonic wavenumber.** `harmonic_wavenumber(1.18424, 1.007825, 34.968853)` = 2991 ± 3; `harmonic_wavenumber(4 * 1.18424, 1.007825, 34.968853) / harmonic_wavenumber(1.18424, 1.007825, 34.968853)` = 2 ± 0.001 | Too big by 2π; The force constant is still per Å²; You multiplied by 1e-20; The reduced mass is still in amu; Too big by 100; μ is the reduced mass, m₁m₂/(m₁ + m₂), not m₁ + m₂.; k is still in hartree/Å²; ω is the square root of k/μ |
| `5.4` | **H₂: bond energy and frequency.** `De_fci` = 4.7 ± 0.1; `De_hf` = 3.629 ± 0.1; `nu_fci` = 4409 ± 15; `nu_hf` = 4587 ± 20 | About 18 eV; `De_hf` equals `De_fci`; About 17 eV; `nu_fci` equals `nu_hf` |
| `5.5d` | Discussion: why Hartree–Fock fails: written answer, ≥ 60 characters in the saved notebook | — |
| `6.1` | **Counting determinants.** `n_det_H2` = 784 ± 0; `n_det_O2` = 8178045876000 ± 0 | You added the α and β counts; `math.comb(28, 2)` treats the two electrons as the same kind; You added the α and β counts; `math.comb(28, 16)` puts all 16 electrons in together |
| `6.2` | **O₂ energy with DFT.** `o2_energy(1.21, spin=2)` = -150.2 ± 0.0001; `o2_energy(1.21, spin=0) > o2_energy(1.21, spin=2)` | — |
| `6.3` | **O₂ triplet and singlet curves.** `len(E_trip) == len(R_grid3) and len(E_sing) == len(R_grid3)`; `np.all(np.asarray(E_sing) > np.asarray(E_trip))`; `(E_sing[np.argmin(E_trip)] - np.min(E_trip)) * 27.2114` = 1.71 ± 0.2 | — |
| `6.4` | **O₂ frequency and bond energy.** `nu_O2` = 1697 ± 30; `De_O2` = 5.53 ± 0.15 | Far too high; Far too large |
| `6.5d` | Discussion: O₂ magnetism and IR: written answer, ≥ 60 characters in the saved notebook | — |
| `7c` | Conclusions: written answer, ≥ 300 characters in the saved notebook | — |
| `8d` | Reflection: written answer, ≥ 60 characters in the saved notebook | — |

Factor diagnoses for every numeric test: ×27.2114, ×0.529177, ×1000, ×6.28319, and a sign flip.

### Lab 2 (`comp02_CO_ab_initio`): ab initio theory and basis sets
13 checks (7 written). The ladder HF → MP2 → CCSD → CCSD(T) at one geometry, then curves, basis sets and the dipole.

- Correlation energy at 1.128 Å, cc-pVDZ: MP2 −7.91, CCSD −8.11, CCSD(T) −8.40 eV (each method under 0.1 s here).
- `fit_minimum` is tested on a synthetic harmonic curve (k = 120 eV/Å² → 2181.6 cm⁻¹ for CO's reduced mass), so a units error
  is caught before any expensive calculation.
- cc-pVTZ constants: HF 1.1045 Å, 148.3 eV/Å², 2425 cm⁻¹; MP2 1.1353 Å, 2138 cm⁻¹; CCSD(T) 1.1325 Å, 2171 cm⁻¹ (expt 1.1283 Å,
  2169.8 cm⁻¹). CCSD(T)/cc-pVDZ: 1.1441 Å, 2146 cm⁻¹.
- Dipole at 1.128 Å, cc-pVTZ: HF −0.245 D (wrong sign), B3LYP +0.119, CCSD +0.180 (expt +0.110, C⁻O⁺). B3LYP μ(1.10) = +0.216,
  μ(1.16) = +0.007 D, dμ/dr = −3.49 D/Å.
- The last cell saves the 29-point CCSD(T) curve and B3LYP dipole curve for Lab 3 (30 s on 8 cores).
- *Teaching points:* the correlation energy is small on the scale of the total energy but as large as the chemistry; MP2 and CC
  are not variational; the near-perfect CCSD(T)/TZ frequency is partly cancellation (r_e still 0.004 Å long; QZ and core
  correlation shorten it): agreement is not convergence. HF's wrong-sign dipole shows how delicate small properties are. The
  CCSD dipole uses the unrelaxed density; the relaxed (energy-derivative) dipole differs slightly.
- *Optional:* `extra_bases = ["cc-pvqz"]` in Question 3.1 adds QZ (about 20 s per point on 2 cores).

**Checks** and the typical mistakes they recognise

| check | what is tested (reference) | typical mistakes recognised |
|---|---|---|
| `1.1` | **Energies up the ladder.** `co_energy(1.128, 'hf', 'sto-3g')` = -3027 ± 0.001; `co_energy(1.128, 'mp2', 'sto-3g')` = -3030 ± 0.001; `co_energy(1.128, 'ccsd(t)', 'sto-3g')` = -3030 ± 0.001 | Your MP2 energy equals the Hartree–Fock energy; That is just the MP2 correlation energy, or close to it; This is the CCSD energy; Far too large |
| `1.2` | **Correlation energies.** `E_corr['hf']` = 0 ± 1e-09; `E_corr['ccsd(t)']` = -8.401 ± 0.01; `E_corr['mp2']` = -7.912 ± 0.01 | That is the total CCSD(T) energy |
| `1.3d` | Discussion: correlation energy: written answer, ≥ 60 characters in the saved notebook | — |
| `2.1` | **Bond length, force constant and wavenumber from a curve.** `fm[0]` = 1.13 ± 0.0001; `fm[1]` = 120 ± 0.01; `fm[2]` = 2182 ± 0.5 | 60 is the coefficient of x², which is k/2; Zero; Off by a factor of 10.2 = 1/√ACC; Check the mass |
| `2.2` | **Curves by method (cc-pVTZ).** `all(m in results for m in ('hf', 'mp2', 'ccsd(t)'))`; `results['ccsd(t)'][0]` = 1.132 ± 0.002; `results['ccsd(t)'][2]` = 2171 ± 5; `results['hf'][2]` = 2425 ± 5; `results['mp2'][2]` = 2138 ± 5 | Close, but not the cc-pVTZ value; Your MP2 result equals Hartree–Fock |
| `2.3d` | Discussion: correlation and the bond: written answer, ≥ 60 characters in the saved notebook | — |
| `3.1` | **Basis-set convergence.** `conv['cc-pvdz'][2]` = 2146 ± 5; `conv['cc-pvdz'][0]` = 1.144 ± 0.002 | This is the cc-pVTZ result again; That looks like a Hartree–Fock frequency |
| `3.2d` | Discussion: agreement vs convergence: written answer, ≥ 60 characters in the saved notebook | — |
| `4.2` | **Dipole derivative.** `mu_plus` = 0.006794 ± 0.005; `mu_minus` = 0.2159 ± 0.005; `dmu_dr` = -3.486 ± 0.03 | You have the difference μ(1.16) − μ(1.10) but haven't divided by the step, 1.16 ; Twice too large |
| `4.3d` | Discussion: dipole and IR intensity: written answer, ≥ 60 characters in the saved notebook | — |
| `abstract` | Abstract: written answer, ≥ 400 characters in the saved notebook | — |
| `6c` | Conclusions: written answer, ≥ 300 characters in the saved notebook | — |
| `7d` | Reflection: written answer, ≥ 60 characters in the saved notebook | — |

Factor diagnoses for every numeric test: ×27.2114, ×6.28319, ×1000, and a sign flip.

### Lab 3 (`comp03_CO_vibrations`): vibrational levels
10 checks (5 written). The finite-difference Hamiltonian is built by the student and tested on a harmonic oscillator first.

- Harmonic test: E_v/ħω = 0.49998, 1.4999, 2.4998, 3.4995.
- On the CCSD(T)/cc-pVTZ curve: fundamental 2145.4 cm⁻¹ (expt 2143.3); ZPE 1082.7 cm⁻¹ measured from the bottom of the spline
  (expt 1081.6); ω_e = 2171.4, ω_e x_e = 12.99 cm⁻¹ (expt 2169.8, 13.29).
- |⟨0|μ|2⟩|²/|⟨0|μ|1⟩|² = 2.35 × 10⁻³; with the factor ν̃₀₂/ν̃₀₁ the overtone is ≈ 1/210 of the fundamental (expt ≈ 1/150).
- ¹³C¹⁶O fundamental 2098.2 cm⁻¹, shift −47.3 cm⁻¹ (expt 2096.1, −47.2); harmonic estimate −47.8.
- *Teaching points:* the Schrödinger equation as a matrix eigenvalue problem; tunnelling past the turning points; the
  correspondence principle at high v; wavefunctions leaning outward (B_v decreases with v); electrical vs mechanical
  anharmonicity; Born–Oppenheimer and isotopes.
- *Note:* the ZPE must be measured from the bottom of the spline (`V_min = V(r).min()`), not from the lowest computed point,
  which lies 0.003 eV (27 cm⁻¹) higher; the check recognises that mistake.

**Checks** and the typical mistakes they recognise

| check | what is tested (reference) | typical mistakes recognised |
|---|---|---|
| `1.1` | **Finite-difference Hamiltonian.** `E_harm[0] / hw` = 0.5 ± 0.002; `E_harm[3] / hw` = 3.5 ± 0.005 | Negative energies; About 10× too small; √2 too large; √2 too small; Level 3 equals level 0 |
| `1.2` | **CO levels and zero-point energy.** `wn[1]` = 2145 ± 0.5; `zpe` = 1083 ± 0.5 | That is E₁ measured from the bottom of the well; Zero; Close |
| `2.1d` | Discussion: wavefunctions: written answer, ≥ 60 characters in the saved notebook | — |
| `3.1` | **ω_e and ω_e x_e.** `we` = 2171 ± 0.5; `wexe` = 12.99 ± 0.05; `abs((we - 2 * wexe) - wn[1]) < 3` | That is the x² coefficient; That is the constant term c0; That is ω_e |
| `4.1` | **Transition dipoles.** `abs(M[1])` = 0.1178 (rel. 0.001); `ratio_21` = 0.002351 (rel. 0.02) | Too large by 1/h; That is ⟨0\|μ\|0⟩, the average dipole in v = 0; Square the transition dipoles; Upside down |
| `4.2d` | Discussion: overtone intensity: written answer, ≥ 60 characters in the saved notebook | — |
| `5.1` | **Isotope shift.** `mu13` = 7.172 ± 1e-05; `f13` = 2098 ± 0.3 | That is the mass of ¹³C alone; That is the total mass; That is the ¹²C¹⁶O fundamental |
| `abstract` | Abstract: written answer, ≥ 400 characters in the saved notebook | — |
| `6c` | Conclusions: written answer, ≥ 300 characters in the saved notebook | — |
| `7d` | Reflection: written answer, ≥ 60 characters in the saved notebook | — |

Factor diagnoses for every numeric test: ×8065.54, ×1000, and a sign flip.

### Lab 4 (`comp04_CO2_normal_modes`): normal modes and IR
12 checks (6 written).

- r_e = 1.1631 Å (expt 1.1600). Analytic H[C_z, C_z] = 195.29 eV/Å²; the finite difference gives 196.18 eV/Å² (0.45 % higher:
  PySCF's analytic Hessian omits the motion of the DFT integration grid, which the finite difference includes). The check
  allows 1.5 eV/Å².
- Nine eigenvalues: five near zero (−21 to +23 cm⁻¹, grid noise), then 653.7 (×2), 1389.0, 2466.9 cm⁻¹ (expt 667, ≈1333, 2349).
- Intensities 2 × 32.9, 0.0, 613.5 km/mol (expt bend 2 × 24, ν₃ 548).
- Earth's 288 K emission peaks at 565 cm⁻¹ (17.7 µm) in the wavenumber representation.
- *Teaching points:* the symmetry selection rule; why two bends; the light carbon atom in ν₃; the bend under the Planck curve;
  band saturation and logarithmic forcing; the ν₁/2ν₂ Fermi resonance (1285/1388 cm⁻¹ Raman doublet) as an extension.

**Checks** and the typical mistakes they recognise

| check | what is tested (reference) | typical mistakes recognised |
|---|---|---|
| `1.1` | **Equilibrium bond length.** `energy_of_r(1.16)` = -188.4 ± 1e-06; `r_e` = 1.163 ± 0.001 | The minimiser stopped at the edge of the search range (1.10 or 1.22 Å) |
| `2.1` | **The Hessian in eV/Å².** `np.shape(H) == (9, 9)`; `H[5, 5]` = 195.3 ± 0.05; `np.allclose(H, H.T, atol=1e-6)` | The units are right, but the rows and columns are in the wrong order |
| `2.2` | **Finite-difference check.** `fd` = 196.2 ± 0.1; `abs(fd - H[5, 5]) < 1.5` | Divide by h², not h; Wrong sign |
| `3.1` | **Normal-mode wavenumbers.** `wn` = [653.7, 653.7, 1389, 2467] ± 1.0 | The frequencies are far too high; The frequencies are too low; Check the mass weighting |
| `3.2d` | Discussion: the modes: written answer, ≥ 60 characters in the saved notebook | — |
| `4.1` | **IR intensities.** `A` = [32.87, 32.87, 0.001078, 613.5] ± 2.0 | Four times too large; The units factor is missing; Four times too small; The symmetric stretch (mode 2) should have zero intensity |
| `4.2d` | Discussion: selection rules: written answer, ≥ 60 characters in the saved notebook | — |
| `5.1` | **The Earth's emission peak.** `nu_peak` = 565 ± 2 | That is an index (a position in the array); The curve peaks at the very start of the range; The curve keeps rising; Check the power |
| `5.2d` | Discussion: the greenhouse effect: written answer, ≥ 60 characters in the saved notebook | — |
| `abstract` | Abstract: written answer, ≥ 400 characters in the saved notebook | — |
| `7c` | Conclusions: written answer, ≥ 300 characters in the saved notebook | — |
| `8d` | Reflection: written answer, ≥ 60 characters in the saved notebook | — |

Factor diagnoses for every numeric test: ×27.2114, ×97.1744, ×0.529177, ×0.280028, and a sign flip.

### Lab 5 (`comp05_CO2_ML_surface`): a machine-learned surface
11 checks (5 written). Students compute 100 test points (25 s on 8 cores, about a minute on 2; `N_TEST = 30` if short of time);
training uses the 320 provided points, so everyone has the same training set and an honest held-out test.

- The sampler is checked by projecting the geometry back onto the normal modes: it must hold exactly the energy it was given.
- Fitted kernel ≈ 383² RBF(ℓ = 12.6, 12.6, 228) + 10⁻⁶ white noise. Test RMS 0.022 meV (0.17 cm⁻¹); mean GP σ 0.165 meV (the
  GP is cautious, about 8× its actual error). Learning curve 0.21, 0.13, 0.038, 0.027, 0.022 meV for 20–320 points.
- GP harmonic frequencies − DFT: +0.1 (bend), −0.5 (ν₁), +4.9 cm⁻¹ (ν₃). The ν₃ difference matches the 0.45 % grid-motion term
  of Lab 4: the GP was fitted to energies, so it agrees with the finite-difference Hessian, not the analytic one.
- Dipole model c = (−0.2714, −0.1549, +0.6024, +0.0340); test RMS 0.0134 D (RMS dipole 0.47 D).
- *Teaching points:* invariance and permutation symmetry in the features; cos θ rather than θ; test vs training error;
  extrapolation (the bond-stretch cut); charge flux explaining the strong ν₃.
- *Implementation note:* the WhiteKernel lower bound of 10⁻⁶ is deliberate. With a smaller bound the GP is ill-conditioned and
  its µeV round-off noise, amplified by finite-difference forces, shows up as energy drift in Lab 6.

**Checks** and the typical mistakes they recognise

| check | what is tested (reference) | typical mistakes recognised |
|---|---|---|
| `1.1` | **Sampling along the normal modes.** `E_harm / E_expect` = 1 (rel. 1e-06) | The geometry holds about 100 times too much energy; Half the energy; `E_tot` should be between 0 and `E_max`; Twice the energy |
| `1.2` | **Your test set.** `E_test.min() > -1e-3`; `E_test.max() < 0.9 and 0.15 < E_test.mean() < 0.45` | These are total energies (about −5000 eV); These energies are far too large for 0.6 eV of vibrational energy |
| `2.1` | **Symmetry-adapted features.** `features(X0)` = [1.163, 1.163, -1] ± 1e-06; `features(np.array([[1.0, 0, 0], [0, 0, 0], [0, 2.0, 0]]))` = [1, 2, 0] ± 1e-09 | cos θ = +1 for the linear molecule; \|cos θ\| > 1 |
| `3.1` | **Gaussian process fit.** `_asym < 1e-5`; `rms` = 0.02153 (rel. 0.5) | Your surface gives different energies for (r₁, r₂) and (r₂, r₁); Far too small; Much larger than expected |
| `5.1d` | Discussion: learning curve and extrapolation: written answer, ≥ 60 characters in the saved notebook | — |
| `6.1` | **Harmonic frequencies of the GP surface.** `wn_gp` = [653.8, 653.8, 1388, 2472] ± 3; `np.max(np.abs(wn_gp - wn_dft)) < 10` | These are the five near-zero eigenvalues (translations and rotations); About 10× too low |
| `7.1` | **Bond-charge dipole model.** `c_dip[0]` = -0.2714 ± 0.003; `rms_mu` = 0.01339 (rel. 0.3) | √3 too small; Too small |
| `7.2d` | Discussion: charges and charge flux: written answer, ≥ 60 characters in the saved notebook | — |
| `abstract` | Abstract: written answer, ≥ 400 characters in the saved notebook | — |
| `9c` | Conclusions: written answer, ≥ 300 characters in the saved notebook | — |
| `10d` | Reflection: written answer, ≥ 60 characters in the saved notebook | — |

Factor diagnoses for every numeric test: ×1000, ×27.2114, and a sign flip.

### Lab 6 (`comp06_CO2_MD_IR`): molecular dynamics and the IR spectrum
13 checks (7 written).

- Forces: max |F| at the minimum 9 × 10⁻⁴ eV/Å; total force ~10⁻⁵ eV/Å at a distorted geometry.
- 200 fs test of Verlet: energy constant to 0.43 meV.
- Time-step study (1 ps, 300 K): fluctuation 43, 3.6, 0.90, 0.27, 0.11 meV for Δt = 4, 2, 1, 0.5, 0.25 fs; slope over 0.5–2 fs
  1.87 (≈ 2: Verlet is second order). The flattening at the smallest step is the GP round-off floor.
- One 10 ps trajectory: bend 651 cm⁻¹, ν₃ 2481 cm⁻¹, symmetric stretch at 2 × 10⁻⁹ relative intensity (absent).
- The `ir_spectrum` check uses synthetic signals: a pure 1000 cm⁻¹ oscillation must peak at 1000 cm⁻¹, and equal oscillations
  at 500 and 1500 cm⁻¹ must give peaks in the ratio 9 (the ν̃² factor).
- 24 × 4 ps averages: ν₃ shows P and R branches about 20 cm⁻¹ apart (classical estimate 2.36√(BT) = 26 cm⁻¹). At 1500 K the bands
  broaden and red-shift strongly.
- *Teaching points:* selection rules survive anharmonic dynamics (they come from symmetry); classical MD has no zero-point energy,
  so at low T it gives near-harmonic positions, not quantum fundamentals (compare Lab 3); rotation shapes bands; the 1500 K
  run approaches the edge of the 0.6 eV sampling of Lab 5.
- *Extension for strong students:* the analytic GP gradient (chain rule through the features), or a Langevin thermostat.

**Checks** and the typical mistakes they recognise

| check | what is tested (reference) | typical mistakes recognised |
|---|---|---|
| `1.1` | **Forces from the surface.** `np.shape(_F) == (3, 3)`; `np.allclose(_F, _ref, atol=1e-4)`; `np.abs(_F.sum(0)).max() < 1e-3 and np.abs(forces(X0)).max() < 0.01` | Wrong sign; Twice too large; Half the right size; Close but not exact |
| `2.1` | **Velocity Verlet.** `np.ptp(_tr[:, 0, 2]) > 1e-3`; `1000 * np.ptp(_et) < 1.0` | The atoms never move; The total energy isn't conserved |
| `3.1` | **Time-step error.** `slope` = 1.866 ± 0.3 | Upside down; Take the logarithms of both |
| `3.2d` | Discussion: the time step: written answer, ≥ 60 characters in the saved notebook | — |
| `4.1d` | Discussion: the trajectory: written answer, ≥ 60 characters in the saved notebook | — |
| `5.1` | **IR spectrum from the dipole.** `peak` = 1000 ± 1.5; `ratio` = 9 (rel. 0.05) | These are frequencies in fs⁻¹, not wavenumbers; Half the right value; The two peaks are equal; The ratio is 3, not 9 |
| `5.2` | **Band positions from MD.** `peaks['bend']` = 651.3 ± 4; `peaks['asymmetric stretch']` = 2481 ± 4 | That is an index (a position in the window); That looks like an intensity, not a wavenumber |
| `5.3d` | Discussion: MD vs harmonic: written answer, ≥ 60 characters in the saved notebook | — |
| `6.1` | **P–R branch separation.** `dPR_expected` = 25.53 ± 0.1 | Take the square root of B·T |
| `7.1d` | Discussion: a hot molecule: written answer, ≥ 60 characters in the saved notebook | — |
| `abstract` | Abstract: written answer, ≥ 400 characters in the saved notebook | — |
| `9c` | Conclusions: written answer, ≥ 300 characters in the saved notebook | — |
| `10d` | Reflection: written answer, ≥ 60 characters in the saved notebook | — |

Factor diagnoses for every numeric test: ×1000, and a sign flip.

### Lab 7 (`comp07_project`): the project
12 checks (5 written). The checks test physics that holds for every molecule: the energy is in eV and the analytic gradient
matches a finite difference; the optimisation converged; the Hessian is symmetric, in the right units, and gives 3N − 5 or
3N − 6 vibrations; the sampler holds the right energy; the swapped copies are used if and only if the outer atoms are identical;
the test error and dipole model are sensible; the forces match a finite difference and MD conserves energy; Figure 1 was saved.

- N₂O reference (the instructor notebook): r(NN) 1.130, r(NO) 1.182 Å (4 BFGS steps); 606 (×2), 1380, 2418 cm⁻¹
  (expt 589, 1285, 2224); 2 × 12, 50, 340 km/mol; 250 DFT points in 72 s (8 cores); test RMS 0.039 meV; GP harmonics within
  1 cm⁻¹; dipole RMS 0.020 D; 2 ps energy fluctuation 0.69 meV; MD 300 K peaks 607, 1380, 2404 cm⁻¹. B3LYP/def2-SVP
  overestimates ν₃ by 9 % and gives a dipole of only 0.006 D (expt 0.16 D): a known difficulty of DFT for N₂O and a good
  discussion point.
- From the earlier version of this lab (2 cores): CS₂ r = 1.562 Å; 405 (×2), 683, 1585 cm⁻¹ (expt 397, 658, 1535);
  2 × 1.2, 0, 528 km/mol. H₂O r = 0.967 Å, 103.1°; 1639, 3792, 3887 cm⁻¹ (expt 1595, 3657, 3756); 56, 5, 27 km/mol.
  Compute on 2 cores: N₂O ≈ 5.5 min, H₂O ≈ 7 min, CS₂ and SO₂ ≈ 9 min.
- To run the instructor notebook for another molecule, set `LAB_SYSTEM=H2O` (etc.) in the environment before
  `python tools/build_notebooks.py --only comp07 --execute`.
- What to look for in student work is in the instructor cell at the end of the notebook.

**Checks** and the typical mistakes they recognise

| check | what is tested (reference) | typical mistakes recognised |
|---|---|---|
| `0.1` | **Your molecule.** `SYSTEM in SYSTEMS` | Check the capitals; `SYSTEM` must be a string |
| `1.1` | **Energy, gradient and geometry.** `_e / _eref` = 1 (rel. 1e-09); `_g[2] / _fd` = 1 (rel. 0.01); `np.abs(res.jac).max() < 0.01` | — |
| `2.1` | **Harmonic analysis and IR intensities.** `np.shape(H) == (9, 9) and np.allclose(H, H.T, atol=1e-6)`; `300 < wn.max() < 4500`; `n_modes == (4 if LINEAR else 3)`; `np.all(A >= 0) and A.max() > 1` | All frequencies are far too low; The frequencies are too high; Too many vibrations |
| `3.1` | **Sampling.** `E_harm / E_expect` = 1 (rel. 1e-06) | About 100 times too much energy; Half the energy |
| `4.1` | **Learned surface and dipole.** `symmetric or len(Fs) == len(F_tr)`; `(not symmetric) or _asym < 1e-5`; `rms < 2.0`; `c_dip.size == (4 if symmetric else 8) and rms_mu < 0.1` | — |
| `5.1` | **Molecular dynamics.** `np.allclose(_Fm, _ref, atol=1e-4)`; `1000 * np.ptp(et) < 5` | Wrong sign |
| `6.1` | **Figure saved.** `os.path.exists(f'{SYSTEM}_figure1.png')` | — |
| `7.1t` | Results table: written answer, ≥ 300 characters in the saved notebook | — |
| `7.2d` | Discussion: written answer, ≥ 100 characters in the saved notebook | — |
| `abstract` | Abstract: written answer, ≥ 600 characters in the saved notebook | — |
| `8c` | Conclusions: written answer, ≥ 300 characters in the saved notebook | — |
| `9d` | Reflection: written answer, ≥ 60 characters in the saved notebook | — |

Factor diagnoses for every numeric test: ×27.2114, ×0.529177, and a sign flip.

## 6. Assessment

Suggested weights (also in the student handout): Labs 1–6 40 %, the project lab 25 %, the project abstract 20 %, figures 15 %.
`lab.score()` and the submission give each lab's check score; the written answers arrive with the submission and in the PDF.

**Lab abstracts and conclusions (each lab).** Look for the purpose, the methods named, the student's own numbers with the
reference beside them, and a conclusion that follows from the numbers. The sample answers in the instructor notebooks show the
expected level.

**Project abstract rubric (20 points)**

| | 4 | 2 | 0 |
|---|---|---|---|
| Question and motivation | clear in one sentence | implied | missing |
| Methods | method, basis, ML model with test error, MD conditions | partly | missing |
| Quantitative results | frequencies and intensities with units, comparison with experiment | qualitative only | missing |
| Physical explanation | IR activity from symmetry/dμ/dQ; harmonic vs MD vs experiment explained | partial | missing |
| Writing | 150–250 words, self-contained, correct units | minor issues | over length or unreadable |

**Figures rubric (15 points):** correct content (5), axis labels with units and legible text (4), a caption that can be read on
its own (4), one clear message per figure (2).

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `ModuleNotFoundError: labkit` | the lab folder was moved out of `comp_chem/`; keep the folder structure (labkit is found one or two folders up) |
| `ModuleNotFoundError: init_access` or no quiz | `shared-public/` not found one to three folders up, or out of date on the hub |
| a check says a variable "doesn't exist yet" | the student skipped a cell, or restarted the kernel; *Run All Above* |
| a written-answer check fails although the cell is filled | the notebook isn't saved (the check reads the saved file); or the student typed into the question cell instead of the cell below |
| 3-D viewer is blank | py3Dmol not installed (the matplotlib fallback should appear), or JavaScript blocked; reload |
| SCF does not converge | a sampled geometry is too distorted; use a smaller `E_MAX`, or skip the point |
| MD energy grows steadily | time step too large for the fastest mode, or forces with the wrong sign or units (the Lab 6/7 checks catch both) |
| MD spectrum empty or at the wrong frequency | dipole not mean-subtracted, frequency axis not divided by c, or the wrong saved time step |
| "using the reference copy in data/" | the earlier lab's result is missing from `comp_chem/results/`; harmless, the lab continues |
| everything is slow | `OMP_NUM_THREADS` too high on a shared node, or several students on one node; stagger the sampling steps |
| GP `ConvergenceWarning`s about kernel bounds | harmless; check the test error |

## 8. Maintaining the material

Edit `src/compNN_*.py` (and the plain key and quiz in `src/compNN_*/`) and rebuild with `python tools/build_notebooks.py`
(add `--only compNN` for one lab, `--execute` to refresh the instructor outputs). Never edit the lab folders or `instructor/`
directly: they are generated.

Source markers: a line ending `# SOLUTION` becomes `x = ...` (or `return ...`) for students; `# %% [markdown] answer` cells
become `ANSWER` for students and a sample answer for instructors; `# %% [markdown] instructor` cells appear only in the
instructor notebook; the first line `# LAB` makes the source a course-format lab folder.

The fallback data in `data/` came from reference runs: `co_ccsdt_curve.npz` (Lab 2's last cell), `co2_harmonic.npz` (Lab 4)
and `co2_b3lyp_samples.npz` (320 sampled points, as in Lab 5).

After changing a computational step, re-run the lab and update the affected answer values in its key (each test's value can be
printed from the executed notebook). A wrong answer that students make often deserves a `wrong` entry with its own feedback;
test it by making the mistake yourself in the instructor notebook and running the check.
