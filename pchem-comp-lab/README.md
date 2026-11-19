# From electrons to infrared spectra: a computational physical chemistry lab

An undergraduate computational laboratory for physical chemistry juniors and seniors. Students start with one DFT
calculation on O₂ and finish by running molecular dynamics on a machine-learned potential energy surface, which they use
to explain why CO₂ absorbs infrared light. Everything runs in Jupyter with pip-installable packages:
[PySCF](https://pyscf.org), NumPy, SciPy, Matplotlib, scikit-learn and py3Dmol.

**Format:** two 3-hour lab sessions plus one week of independent project work, ending in an abstract and figures.

| # | notebook | session | what students build | physical idea |
|---|---|---|---|---|
| 1 | `comp01_H2_O2` (course format) | 1 | H radial functions → Gaussian (STO-nG) fits → variational energies (H, He) → H₂⁺ LCAO and accurate orbitals → H₂ → O₂ with DFT; energy functions, curves, force constants; abstract and conclusions | the nuclear cusp and Gaussian basis sets; the variational principle (Hylleraas); the one-electron bond; electron correlation; why FCI is impossible for O₂; triplet O₂; why O₂ is IR-inactive |
| 2 | `02_ab_initio_CO` | 1 | HF → MP2 → CCSD(T) curves; basis-set convergence; dipole | electron correlation; systematic improvability; dμ/dr |
| 3 | `03_vibrational_levels` | 1 | finite-difference Schrödinger solver | anharmonicity, zero-point energy, overtones, isotope shifts |
| 4 | `04_CO2_harmonic_IR` | 2 | mass-weighted Hessian, normal modes, IR intensities | selection rules from symmetry; the greenhouse bend |
| 5 | `05_ML_potential_CO2` | 2 | sampler, symmetry-adapted features, Gaussian-process surface, dipole model | learning a surface; testing on unseen data |
| 6 | `06_MD_IR_CO2` | 2 | forces, velocity Verlet, IR spectrum from the dipole | classical dynamics; spectrum = Fourier transform of μ(t); rotational P/R contours |
| 7 | `07_project_template` | project week | the same workflow for N₂O, OCS, HCN, CS₂, H₂O or SO₂ | abstract + figures |

## Repository layout

Notebook 1 is in the CHEM 3376 course format, the same as the conductance lab: it lives in its own lab folder
(`comp01_H2_O2/`, with its answer checker `pchem_check.py`, the answer key `answers_comp01.json` (encoded in the student folder) and a quiz file) and starts with the course
initialization cell, which loads `init.py` or `init_access.py` from `shared-public/`. Copy the lab folder into the course
folder next to `shared-public/`, like `lab05/`. (The init cell also looks two and three levels up, so the repository can
sit inside the course folder.) Notebooks 2–7 are still in the earlier format and will be converted the same way.

```
comp01_H2_O2/  Notebook 1, course format: notebook, answer checks, quiz   <- what students open
notebooks/     student notebooks 2-7 (solutions removed)
instructor/    the same notebooks with solutions, sample answers and executed outputs
               (instructor/comp01_H2_O2/ for Notebook 1)
labkit/        small helper package: units, 3-D viewers/animations, fallback data loaders
data/          precomputed fallback data (CO curve, CO2 harmonic results, 320 CO2 DFT points)
docs/          student handout and instructor guide
src/           single source for each notebook (edit these, then rebuild)
tools/         build_notebooks.py: makes notebooks/ and instructor/ from src/
```

## Setup on a JupyterHub

```bash
pip install -r requirements.txt
```

Students need the `notebooks/`, `labkit/` and `data/` folders. The notebooks import `labkit` from the folder above
(`sys.path.insert(0, "..")`), so keep the folder structure. **Do not distribute `instructor/`** to students; for a public
repository, keep the instructor notebooks in a separate private repository (see the instructor guide).

An [nbgitpuller](https://nbgitpuller.readthedocs.io) link gives each student a copy with one click.

On a shared hub, limit PySCF's threads so that one student does not take every core: set `OMP_NUM_THREADS=2`
(or 4) in the hub's environment.

## Rebuilding the notebooks

Edit the sources in `src/`. Solutions are marked with `### BEGIN SOLUTION` / `### END SOLUTION`, and instructor-only
cells start with `# %% [markdown] instructor` or `# %% instructor`. Then:

```bash
python tools/build_notebooks.py             # writes notebooks/ and instructor/
python tools/build_notebooks.py --execute   # also runs every instructor notebook (about 15 minutes)
```

## Reference run times

Executed on 2 CPU cores (PySCF 2.15, scikit-learn 1.9), a typical per-student allocation on a hub. Students spend most of
the session reading, writing and thinking; the compute time is a small part of it.

| notebook | 01 | 02 | 03 | 04 | 05 | 06 | 07 (N₂O) |
|---|---|---|---|---|---|---|---|
| compute time | 1 min | 2 min | 2 s | 25 s | 1.5 min | 3 min | 5.5 min (5–10 min by molecule) |
