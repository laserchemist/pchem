# Instructor guide
### From electrons to infrared spectra: a computational physical chemistry lab

## 1. Purpose and learning goals

The lab takes physical chemistry juniors and seniors from a single electronic-structure calculation to a dynamical simulation
of an infrared spectrum, with the physics in front and the code kept short. By the end, students should be able to:

1. explain the Born–Oppenheimer potential energy surface and compute one (DFT and ab initio), including spin states;
2. explain electron correlation and basis-set convergence, and why wavefunction methods can be improved systematically;
3. obtain harmonic frequencies from curvature and normal modes from the mass-weighted Hessian;
4. solve the 1-D vibrational Schrödinger equation on a grid, and relate anharmonicity, zero-point energy and overtones to
   the shape of the curve;
5. predict IR activity from symmetry and from dμ/dQ, and connect the CO₂ bend to the greenhouse effect;
6. build and *test* a machine-learned potential energy surface (Gaussian process) with symmetry-adapted coordinates;
7. integrate Newton's equations (velocity Verlet), choose a time step, and compute a spectrum from the Fourier transform of
   the dipole moment;
8. communicate a computational result as an abstract and figures.

The ML/MD part follows the practice of our research surfaces, in a form undergraduates can complete: symmetry built into the
inputs, sampling by normal-mode displacement, held-out test sets, harmonic frequencies checked against the reference at every
fit, a physically motivated dipole model, and classical (not ring-polymer) dynamics.

## 2. Schedule

| | content | approx. time | notes |
|---|---|---|---|
| **Session 1** | Introduction (15 min): PES, DFT vs wavefunction methods, Jupyter basics | | running the first worked example (H atom) together helps |
| | Notebook 1 (`comp01_H2_O2`): H atom → H₂⁺ → H₂ → O₂ | 75 min | 1 min of compute; course format with auto-checks |
| | Notebook 2: CO ab initio and basis sets | 60 min | 2 min of compute |
| | Notebook 3: vibrational levels | 50 min | almost no compute; the most mathematical notebook |
| **Session 2** | Brief recap (10 min): normal modes, IR selection rule | | |
| | Notebook 4: CO₂ harmonic analysis and IR | 50 min | |
| | Notebook 5: ML surface | 60 min | 1–2 min of DFT sampling |
| | Notebook 6: MD and IR spectrum | 60 min | 3 min of MD |
| | Project briefing (10 min): assign molecules, show the abstract example | | |
| **Week** | Notebook 7 project, abstract and figures | 4–6 h of student time | an office hour midway helps |

Students who fall behind in Session 1 can finish Notebook 3 at home; it has no expensive steps. Notebooks 4–6 depend on each
other's saved files, but each loads the precomputed copy from `data/` if the student's file is missing, so nobody is blocked.

**Assign project molecules** so that the class covers several. Good pairings for discussion: N₂O vs CS₂ (both linear, with and
without an inversion centre), H₂O vs SO₂ (both bent; light vs heavy atoms). HCN and H₂O need Δt = 0.25 fs (set automatically).

## 3. Setup

### Packages
Everything installs with pip: `pip install -r requirements.txt` (pyscf, numpy, scipy, matplotlib, scikit-learn, py3Dmol).
PySCF provides Linux and macOS wheels; it does not support native Windows, so students need the hub (or WSL).

### Distributing to students
The student-facing material is `comp01_H2_O2/`, `notebooks/`, `labkit/`, `data/` and `docs/student_handout.md`. The `instructor/` folder
contains the solutions. Two options:

1. **Two repositories (recommended).** A public (or course-visible) repository without `instructor/`, used with an
   nbgitpuller link, and a private repository with everything. Copy the tree, delete `instructor/` from the public copy,
   and push each.
2. **One private repository** used only by staff; copy the student folders into the hub's shared or home directories.

nbgitpuller link format:
`https://<hub>/hub/user-redirect/git-pull?repo=<repo-url>&branch=main&urlpath=lab/tree/<repo-name>/notebooks`

### Server resources
PySCF uses OpenMP and by default takes every core. On a shared hub, set `OMP_NUM_THREADS=2` in the spawner environment.
With 2 cores per student, the compute times in the README apply. Memory: well under 1 GB per student; the largest job is
optional CCSD(T)/cc-pVQZ for CO (~0.5 GB).

### Before the first session
Run `python tools/build_notebooks.py --execute` on the hub itself (about 15 minutes). This checks the installation and gives
you timings for your hardware. Then open one student notebook as a test account and check that the py3Dmol viewer appears
(JupyterLab ≥ 3 and classic Notebook both work; the viewer needs JavaScript enabled).

## 4. Notebook-by-notebook notes and reference values

Reference values below are from the executed instructor notebooks (B3LYP/def2-SVP unless stated). Students' values should
agree closely; random sampling changes the ML and MD numbers slightly.

### Notebook 1 (`comp01_H2_O2`): from the hydrogen atom to O₂
Course format: the initialization cell, 💡 hint dropdowns, `x = ...` blanks (one to four per question), a 6-question
practice quiz (`questionsComp01.json`), a reflection, and `lab.score()` at the end.

**Answer checks.** Instead of gofer, the lab uses its own checker, like the Neighbor Games' `neighbor_check`:
`lab.check('3.1')` after each question (18 checks, 5 of them discussion answers checked for length in the saved
notebook). The engine is `pchem_check.py`, reusable in any lab; this lab's tests are in `checks_comp01.py`.
- Each check stops at the first failing test and shows a hint written for that mistake (e.g. "charge = 1: two protons,
  one electron"), so students can fix the cell and try again. There is no penalty for retrying.
- Every attempt is recorded with a time stamp in `.comp01_check_log.json` next to the notebook, so the record
  survives a kernel restart. `lab.score()` shows each question's status, attempts, and the attempt on which it was first
  correct, with the total (e.g. "16 of 18 correct, 12 on the first try").
- The record is a plain file a determined student could edit; it is meant for feedback and self-monitoring. For grading,
  re-run the notebook, or collect the log files (one per student folder) if you want attempt counts.
- To add or change a check, edit `checks_comp01.py`: each test is a one-line `lambda ns: ...` (ns is the notebook's
  variables) with the hint to show when it is False.
The order builds from systems with exact answers to one without:

- **H atom** (first calculation is a fully worked cell). Errors above −0.5 Eh: STO-3G 909 meV, 6-31G 48, cc-pVDZ 20,
  cc-pVTZ 5.2, cc-pVQZ 1.5 meV. One electron, so HF is exact; the only error is the basis (variational principle).
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

### Notebook 2: CO ab initio
- Correlation energy at cc-pVDZ: MP2 −7.9, CCSD −8.1, CCSD(T) −8.4 eV.
- cc-pVTZ constants: HF 1.105 Å, 2425 cm⁻¹; MP2 1.135 Å, 2138 cm⁻¹; CCSD(T) 1.133 Å, 2171 cm⁻¹ (expt 1.128 Å, 2170).
  CCSD(T)/cc-pVDZ: 1.144 Å, 2146 cm⁻¹.
- Dipole at 1.128 Å (cc-pVTZ): HF −0.245 D (wrong sign), B3LYP +0.119, CCSD +0.180 (expt +0.11, C⁻O⁺). B3LYP dμ/dr ≈ −3.5 D/Å.
- *Teaching point:* the near-perfect CCSD(T)/TZ frequency involves cancellation of errors (r_e is still 0.004 Å long);
  agreement is not convergence. The MP2 `make_rdm1` density is unrelaxed; mention it if a student tries MP2 dipoles.
- The last cell computes the 29-point curve used by Notebook 3 (about 1.5 min).

### Notebook 3: vibrational levels
- Fundamental 2145.4 cm⁻¹ (expt 2143.3); ω_e = 2171.4, ω_e x_e = 13.0 (expt 2169.8, 13.29); ZPE 1055 cm⁻¹.
- |⟨0|μ|2⟩|²/|⟨0|μ|1⟩|² = 2.4 × 10⁻³; with the ν̃ factor the overtone is ≈ 1/200 of the fundamental (expt ≈ 1/150).
- ¹³C¹⁶O shift −47.3 cm⁻¹ (expt −47.2); harmonic estimate −47.8.
- *Common errors:* the kinetic prefactor ħ²·ACC/(2μh²); wavefunction normalisation (divide eigenvectors by √h).

### Notebook 4: CO₂ harmonic analysis
- r_e = 1.1631 Å (expt 1.1600). Hessian element H[C_z, C_z] = 196.18 eV/Å², analytic and finite difference agree.
- Five near-zero eigenvalues (−21 to +23 cm⁻¹: DFT grid noise), then 654 (×2), 1389, 2472 cm⁻¹.
- Intensities 2 × 33, 0, 614 km/mol (expt bend 2 × 24, ν₃ 548).
- *Teaching points:* symmetry selection rule; the bend under the 288 K Planck curve; band saturation and the logarithmic
  forcing; the ν₁/2ν₂ Fermi resonance (1285/1388 cm⁻¹ Raman doublet) as an extension.
- *Common errors:* reshaping the (3,3,3,3) Hessian without the transpose (wrong ordering); forgetting to divide the mass-weighted
  eigenvectors by √m for Cartesian displacements.

### Notebook 5: ML surface
- Students compute 100 test points (≈ 1 min); training uses the 320 provided points, so everyone has the same training set
  and an honest held-out test.
- Test RMS ≈ 0.02 meV (0.16 cm⁻¹); learning curve 0.31 → 0.02 meV from 20 to 320 points; GP harmonic frequencies within
  1 cm⁻¹ of DFT; dipole model c = (−0.271, −0.155, +0.602, +0.034), test RMS 0.013 D.
- The 1–2 Å stretch cut shows the GP following DFT inside the data and drifting outside, with σ growing.
- *Teaching points:* invariance and permutation symmetry in the features; why cos θ rather than θ; test vs training error;
  extrapolation; the dipole model's charge flux explaining the strong ν₃.
- *Implementation note:* the WhiteKernel lower bound of 10⁻⁶ is deliberate. With a smaller bound the GP is ill-conditioned:
  weights ~4 × 10⁴ give ~µeV round-off noise in predictions, which finite-difference forces amplify and which shows up as
  energy drift in MD.

### Notebook 6: MD and the IR spectrum
- Energy fluctuation scales as Δt²: about 0.1 meV at 0.25 fs, 1 meV at 1 fs, several meV at 2 fs; 4 fs fails.
- One 10 ps trajectory at 300 K: bend 653 cm⁻¹, ν₃ 2481 cm⁻¹, symmetric stretch at 10⁻⁹ relative intensity (absent).
- 24 × 4 ps average: ν₃ shows two branches near 2462 and 2479 cm⁻¹ (classical P/R contour; expected separation ≈ 26 cm⁻¹).
  At 1500 K the bands broaden and red-shift strongly (ν₃ spreads over 2320–2450 cm⁻¹).
- *Teaching points:* symmetry selection rules survive anharmonic dynamics; classical MD has no zero-point energy, so it gives
  near-harmonic positions at low T, not quantum fundamentals (compare Notebook 3); rotation shapes bands.
- *Extension for strong students:* replace the finite-difference forces by the analytic gradient of the GP (chain rule through
  the features), or add a Langevin thermostat.

### Notebook 7: projects
- N₂O reference (instructor notebook): r(NN) 1.130, r(NO) 1.182 Å; 606 (×2), 1380, 2418 cm⁻¹; 2 × 12, 50, 340 km/mol;
  ML test RMS 0.04 meV, GP harmonics within 1 cm⁻¹. B3LYP/def2-SVP overestimates ν₃ by 9 % (2418 vs 2224): a known
  difficulty of DFT for N₂O and a good point for discussion. The DFT dipole at equilibrium is nearly zero (expt 0.16 D).
- CS₂ (tested): r = 1.562 Å; 405 (×2), 683, 1585 cm⁻¹ (expt 397, 658, 1535); 2 × 1.2, 0, 528 km/mol; ML test 0.04 meV.
  Same symmetry as CO₂: ν₁ inactive, and a much weaker bend.
- H₂O (tested): r = 0.967 Å, 103.1°; 1639, 3792, 3887 cm⁻¹ (expt 1595, 3657, 3756); 56, 5, 27 km/mol; ML test 0.11 meV,
  GP harmonics within 4 cm⁻¹. All three modes IR-active; the large permanent dipole (≈ 2 D) makes a strong rotational band.
- Compute on 2 cores: N₂O ≈ 5.5 min, H₂O ≈ 7 min, CS₂ and SO₂ ≈ 9 min (more electrons; most of it is the 250 DFT points).
- The pure-rotation band at low wavenumber (polar molecules) is cut off by starting the plot at 200 cm⁻¹; curious students who
  extend the range will find it.

## 5. Assessment

Suggested weights (also in the student handout): notebooks 1–6 40 %, project notebook 25 %, abstract 20 %, figures 15 %.

**Abstract rubric (20 points)**

| | 4 | 2 | 0 |
|---|---|---|---|
| Question and motivation | clear in one sentence | implied | missing |
| Methods | method, basis, ML model with test error, MD conditions | partly | missing |
| Quantitative results | frequencies and intensities with units, comparison to experiment | qualitative only | missing |
| Physical explanation | IR activity from symmetry/dμ/dQ, harmonic vs MD vs experiment explained | partial | missing |
| Writing | ≤ 200 words, self-contained, correct units | minor issues | over length or unreadable |

**Figures rubric (15 points):** correct content (5), axis labels with units and legible text (4), caption that can be read on
its own (4), one clear message per figure (2).

## 6. Troubleshooting

| symptom | cause and fix |
|---|---|
| `ModuleNotFoundError: labkit` | the notebook was moved out of `notebooks/`; keep the folder structure or adjust `sys.path` |
| 3-D viewer is blank | py3Dmol not installed (the matplotlib fallback should appear instead), or JavaScript blocked; try a reload |
| SCF does not converge | a sampled geometry is too distorted; use a smaller `E_max`, or skip the point |
| MD energy grows steadily | time step too large for the fastest mode, or forces with the wrong sign or units |
| MD spectrum is empty or at the wrong frequency | dipole not mean-subtracted, or the frequency axis not divided by c, or the wrong saved time step |
| everything is slow | `OMP_NUM_THREADS` too high on a shared node, or several students on one node; stagger the sampling steps |
| GP warnings about bounds | harmless `ConvergenceWarning`s about kernel bounds; the fit is still good (check the test error) |

## 7. Maintaining the material

Edit `src/*.py` and rebuild with `python tools/build_notebooks.py` (add `--execute` to refresh the instructor outputs and timings).
The student notebooks are generated by removing everything between `### BEGIN SOLUTION` and `### END SOLUTION` and all
instructor-only cells, so never edit `notebooks/` or `instructor/` directly.

The fallback data in `data/` came from the instructor run: `co_ccsdt_curve.npz` (Notebook 2's last cell),
`co2_harmonic.npz` (Notebook 4) and `co2_b3lyp_samples.npz` (320 sampled points, as in Notebook 5).
