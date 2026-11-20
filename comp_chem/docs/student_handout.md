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

| when | labs | goals |
|---|---|---|
| **Session 1** (3 h) | 1. From the H atom to O₂ · 2. CO ab initio · 3. Vibrational levels | the variational principle; the chemical bond in H₂⁺ and H₂; electron correlation; spin states of O₂; basis sets; quantum vibrational levels and anharmonicity |
| **Session 2** (3 h) | 4. CO₂ normal modes and IR · 5. ML surface · 6. Molecular dynamics | normal modes; IR selection rules; machine-learned surfaces; classical MD; spectra from dynamics |
| **Project week** | 7. Your project | the full workflow for a new molecule; abstract and figures |

Finish any parts of Labs 1–6 that you did not complete in the lab session before you start your project. Each lab opens from its
button on the course's **Computational Chemistry** page (`compute.html`).

## How to work

- Each lab is one notebook in its own folder (`comp_chem/comp01_H2_O2/`, `comp_chem/comp02_CO_ab_initio/`, ...). Run the
  initialization cell first, then the cells **in order** (Shift + Enter). If something seems wrong, use *Kernel → Restart & Run All*.
- Replace each `...` with your answer; most answers are one line. Stuck? Open the 💡 **Hint** under the question.
- After each question, run its check cell, `lab.check('…')`. A ✅ means you got it; a ❌ tells you what to fix, often naming
  the exact slip (a missing unit conversion, a sign, an index instead of a value). Try again as often as you like: there is no
  penalty for retrying, and `lab.score()` at the end shows your record.
- **Discussion questions** ask you to explain a result. Write your answers in the markdown cell under each question (complete
  sentences, with numbers), then save the notebook and run its check.
- Every lab starts with an **Abstract** (write it last) and ends with **Conclusions**, a **Reflection** and the **Submit** button.
  Then export a PDF and upload it to Canvas.
- Work with your lab partner, but each of you writes and runs your own notebook.
- Each lab saves its results (for example `co2_harmonic.npz`) in `comp_chem/results/` for the next lab. If a lab is
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

## Your project (Lab 7)

Choose **one** molecule: N₂O, OCS, HCN, CS₂, H₂O or SO₂. (Check with your instructor so that the class covers several.)
Apply the full workflow: geometry, harmonic modes and intensities, ML surface with a test-set error, MD at 300 K and one
higher temperature, the MD spectrum, and a comparison with experiment. Lab 7 gives the structure, lists what to change
from your CO₂ code, and checks the physics at each step.

Budget about 4–6 hours of your own time. Most of the computer time is in the training data (1–5 minutes) and the averaged
spectra (2–4 minutes).

### What to hand in

1. **Your completed Labs 1–7**, run from top to bottom, with all questions answered and submitted.
2. **An abstract** of 150–250 words (in the Abstract cell of Lab 7). It should state:
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
> and the asymmetric stretch at 2467 cm⁻¹ (614 km mol⁻¹). A Gaussian-process surface trained on 320 DFT energies reproduces
> held-out energies to 0.02 meV and the harmonic frequencies to within 5 cm⁻¹. Classical molecular dynamics at 300 K on this surface ...*

## Grading (suggested; your instructor may change it)

| part | weight |
|---|---|
| Labs 1–6: answer checks, discussion answers, abstracts and conclusions | 40 % |
| Project (Lab 7): correct workflow, test-set error reported, sensible time step | 25 % |
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
