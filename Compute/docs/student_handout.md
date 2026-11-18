# From electrons to infrared spectra
### Computational laboratory — student handout

## What you will do

Why does CO₂ warm the planet while O₂ and N₂, which make up 99 % of the air, do not? You will answer this question from
first principles. You will start from the Schrödinger equation for the electrons and build up, one function at a
time, to a movie of a vibrating, rotating CO₂ molecule and its infrared spectrum. Then you will do the same for a molecule of
your choice and report the results as a short scientific abstract with figures.

You will write the important pieces of code yourself: the energy calculation, the frequency analysis, the machine-learned
surface, the integrator for Newton's equations, and the spectrum. Each piece is short (5–20 lines). The physics is the
point; the code is how you make the physics do something.

## Schedule

| when | notebooks | goals |
|---|---|---|
| **Session 1** (3 h) | 1. O₂ with DFT · 2. CO ab initio · 3. Vibrational levels | potential energy curves; spin states; electron correlation and basis sets; quantum vibrational levels and anharmonicity |
| **Session 2** (3 h) | 4. CO₂ normal modes and IR · 5. ML surface · 6. Molecular dynamics | normal modes; IR selection rules; machine-learned surfaces; classical MD; spectra from dynamics |
| **Project week** | 7. Your project | the full workflow for a new molecule; abstract and figures |

Finish any parts of Notebooks 1–6 that you did not complete in the lab before you start your project.

## How to work

- Open the notebooks in the `notebooks/` folder and run the cells **in order** (Shift + Enter). If something seems wrong,
  use *Kernel → Restart & Run All*.
- Cells marked `# YOUR CODE HERE` are yours to write. Most are followed by a **check** cell; when the check passes, move on.
- **Questions** in the notebooks ask you to explain a result. Write your answers in the markdown cell provided (complete
  sentences, with numbers). These answers are part of your grade.
- Work with your lab partner, but each of you writes and runs your own notebook.
- Each notebook saves its results (for example `co2_harmonic.npz`) for the next notebook. If a notebook is
  missing its input, it uses a provided copy in `data/` automatically, so you can always continue.
- The 3-D viewers can be rotated with the mouse and zoomed with the scroll wheel.

## Units used in the lab

| quantity | unit | note |
|---|---|---|
| energy | eV (PySCF works in hartree) | 1 Eh = 27.2114 eV; 1 eV = 8065.5 cm⁻¹ |
| length | Å (PySCF's Hessian is per bohr²) | 1 bohr = 0.529177 Å |
| mass | amu | |
| time | fs | |
| wavenumber | cm⁻¹ | ν̃ = ω / (2πc), c = 2.9979 × 10⁻⁵ cm/fs |
| dipole | debye | 1 e·Å = 4.803 D |

The one awkward conversion is Newton's law: a [Å/fs²] = F [eV/Å] / m [amu] × 0.0096485. It is `lk.ACC` in the code.

## Your project (Notebook 7)

Choose **one** molecule: N₂O, OCS, HCN, CS₂, H₂O or SO₂. (Check with your instructor so that the class covers several.)
Apply the full workflow: geometry, harmonic modes and intensities, ML surface with a test-set error, MD at 300 K and one
higher temperature, the MD spectrum, and a comparison with experiment. Notebook 7 gives the structure and lists what to change
from your CO₂ code.

Budget about 4–6 hours of your own time. Most of the computer time is in the training data (1–5 minutes) and the averaged
spectra (2–4 minutes).

### What to hand in

1. **Your completed notebooks 1–7**, run from top to bottom, with all questions answered.
2. **An abstract** of at most 200 words. It should state:
   - the question,
   - what you did (methods, in one or two sentences: B3LYP/def2-SVP, harmonic analysis, Gaussian-process surface with
     its test error, classical MD),
   - the main quantitative results (frequencies, intensities, how MD compares with harmonic and experiment),
   - the physical explanation: which vibrations absorb infrared light, and why.
3. **One or two figures**, each with a caption of 2–4 sentences. Figure 1 is usually the spectrum: MD at two temperatures,
   harmonic sticks, experimental lines. Figure 2 can show the normal modes, a trajectory film strip, or the accuracy of
   your ML surface. Label the axes with units, and make the figures readable at the size they will be printed.

A good abstract can be read without the figures, and a good figure can be read without the abstract.

### Example of the level of detail expected (for CO₂, which you may not use for your project)

> *We computed the infrared spectrum of CO₂ from first principles to identify which vibrations make it a greenhouse gas.
> B3LYP/def2-SVP harmonic analysis gives the bend at 654 cm⁻¹ (2 × 33 km mol⁻¹), the symmetric stretch at 1389 cm⁻¹ (0 km mol⁻¹)
> and the asymmetric stretch at 2472 cm⁻¹ (614 km mol⁻¹). A Gaussian-process surface trained on 320 DFT energies reproduces
> held-out energies to 0.02 meV and the harmonic frequencies to 1 cm⁻¹. Classical molecular dynamics at 300 K on this surface ...*

## Grading (suggested; your instructor may change it)

| part | weight |
|---|---|
| Notebooks 1–6: working code and answered questions | 40 % |
| Project notebook: correct workflow, test-set error reported, sensible time step | 25 % |
| Abstract | 20 % |
| Figures and captions | 15 % |

## Reading

- D. A. McQuarrie and J. D. Simon, *Physical Chemistry: A Molecular Approach*, chapters on vibrational spectroscopy and
  computational chemistry (or the corresponding chapters of your course text).
- C. J. Cramer, *Essentials of Computational Chemistry*, 2nd ed. (Wiley, 2004), chapters 4–8.
- PySCF user guide: <https://pyscf.org/user.html>
- C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning* (MIT Press, 2006), chapter 2, free online at
  <https://gaussianprocess.org/gpml/>.
- NASA Earth Observatory, "The Carbon Cycle" and "Climate and Earth's Energy Budget", for the greenhouse background.
