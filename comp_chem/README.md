# From electrons to infrared spectra: a computational physical chemistry lab

An undergraduate computational laboratory for physical chemistry juniors and seniors. Students start with the hydrogen
atom, where the exact answer is known, and finish by running molecular dynamics on a machine-learned potential energy
surface, which they use to explain why CO₂ absorbs infrared light. Everything runs in Jupyter with pip-installable packages:
[PySCF](https://pyscf.org), NumPy, SciPy, Matplotlib, scikit-learn and py3Dmol.

**Format:** two 3-hour lab sessions plus one week of independent project work, ending in an abstract and figures. Students
open each lab from its button on the course page `compute.html` (in the repository root), which uses nbgitpuller.

All seven labs are in the CHEM 3376 course format: the course initialization cell, 💡 hints, `x = ...` blanks, answer checks
with feedback for typical wrong answers (`lab.check(...)`), discussion questions, a practice quiz, an **Abstract** (written
last) and **Conclusions**, a reflection, `lab.score()` and the course **Submit** button.

| # | lab folder | session | what students build | physical idea | checks |
|---|---|---|---|---|---|
| 1 | `comp01_H2_O2` | 1 | H radial functions → Gaussian (STO-nG) fits → variational energies (H, He) → H₂⁺ LCAO → H₂ → O₂ with DFT | the nuclear cusp and Gaussian basis sets; the variational principle (Hylleraas); the one-electron bond; electron correlation; triplet O₂; why O₂ is IR-inactive | 25 |
| 2 | `comp02_CO_ab_initio` | 1 | HF → MP2 → CCSD(T) energies and curves; basis-set convergence; dipole and dμ/dr | electron correlation; systematic improvability; agreement vs convergence; IR intensity | 13 |
| 3 | `comp03_CO_vibrations` | 1 | finite-difference Schrödinger solver, tested on a harmonic oscillator | zero-point energy, anharmonicity, overtones, isotope shifts | 10 |
| 4 | `comp04_CO2_normal_modes` | 2 | equilibrium geometry, Hessian (checked by finite difference), mass-weighted normal modes, IR intensities, Planck curve | selection rules from symmetry; the greenhouse bend | 12 |
| 5 | `comp05_CO2_ML_surface` | 2 | normal-mode sampler, symmetry-adapted features, Gaussian-process surface, learning curve, bond-charge dipole | learning a surface; testing on unseen data; extrapolation | 11 |
| 6 | `comp06_CO2_MD_IR` | 2 | forces, velocity Verlet, time-step test, IR spectrum from the dipole, trajectory averaging, 1500 K | classical dynamics; spectrum = Fourier transform of μ(t); rotational P/R contours | 13 |
| 7 | `comp07_project` | project week | the same workflow for N₂O, OCS, HCN, CS₂, H₂O or SO₂ (checks test physics, not molecule-specific numbers) | abstract + figures | 12 |

## Repository layout

Each lab lives in its own folder, with its notebook, the answer checker `pchem_check.py`, its answer key
`answers_compNN.json` (encoded in the student folder) and a quiz file `questionsCompNN.json`. The initialization cell loads
`init.py` or `init_access.py` from `shared-public/` (it looks one, two and three folders up, so `pchem/shared-public` is
found). Labs 2–7 import `labkit` from `comp_chem/`, and pass results to the next lab through `comp_chem/results/`
(not in git); a lab whose input is missing uses the reference copy in `data/`.

```
comp01_H2_O2/ ... comp07_project/   the seven labs, course format     <- what students open
instructor/    the same labs with solutions, sample answers, plain answer keys and executed outputs
labkit/        small helper package: units, 3-D viewers/animations, data loaders, results/ helpers
data/          precomputed fallback data (CO curve, CO2 harmonic results, 320 CO2 DFT points)
results/       created when students run the labs: each lab's output for the next (ignored by git)
docs/          student handout and the instructor book (instructor_guide.md)
src/           single source for each lab (compNN_*.py) and its assets (src/compNN_*/: plain answer key, quiz);
               src/_shared/pchem_check.py is the one master checker, copied into every lab folder by the build
tools/         build_notebooks.py: makes the lab folders and instructor/ from src/
```

## Setup on a JupyterHub

```bash
pip install -r requirements.txt
```

Students need the lab folders, `labkit/` and `data/`; keep the folder structure. **Do not distribute `instructor/` or
`src/`** (they hold the solutions and the plain answer keys): nbgitpuller copies the *whole* repository to every student, so
for a public repository keep those two folders in a separate private repository (see the instructor book, §3).

The [nbgitpuller](https://nbgitpuller.readthedocs.io) buttons on `compute.html` give each student a copy with one click.

On a shared hub, limit PySCF's threads so that one student does not take every core: set `OMP_NUM_THREADS=2`
(or 4) in the hub's environment.

## Rebuilding the notebooks

Edit the sources in `src/`. A solution line ends with `# SOLUTION` (students see `x = ...`), sample answers are
`# %% [markdown] answer` cells (students see `ANSWER`), and instructor-only cells start with `# %% [markdown] instructor`.
Then:

```bash
python tools/build_notebooks.py               # writes the lab folders and instructor/
python tools/build_notebooks.py --only comp04 # one lab
python tools/build_notebooks.py --execute     # also runs every instructor notebook (about 10 minutes on 8 cores)
```

The answer keys are edited as plain JSON in `src/compNN_*/answers_compNN.json`; the build encodes the students' copy. The
key format, and how the checks and feedback work, are in the instructor book (§4) and the docstring of `pchem_check.py`.

## Reference run times

Executed on 2 CPU cores (PySCF 2.15, scikit-learn 1.9), a typical per-student allocation on a hub. Students spend most of
the session reading, writing and thinking; the compute time is a small part of it.

| lab | 1 | 2 | 3 | 4 | 5 | 6 | 7 (N₂O) |
|---|---|---|---|---|---|---|---|
| compute time, 2 cores | 1 min | 2 min | 2 s | 25 s | 1.5 min | 3 min | 5.5 min (5–10 min by molecule) |
| full instructor run, 8 cores | 19 s | 45 s | 4 s | 14 s | 36 s | 68 s | 2.5 min |
