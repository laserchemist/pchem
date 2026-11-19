"""
checks_comp01.py — the answer checks for Computational Lab 1 (From the Hydrogen Atom to O₂).

Each test is (function of the notebook's variables → True/False, hint shown if it is False).
Tests run in order and stop at the first failure, so put the most basic test first.
Reference values: PySCF 2.x, instructor run.
"""
import numpy as np

H_ATOM_EXACT = -0.5


def _close(a, b, tol):
    return abs(a - b) < tol


CHECKS = {
    "1.1": {
        "title": "Unit conversions",
        "needs": ["hartree_to_eV", "bohr_to_angstrom"],
        "tests": [
            (lambda ns: _close(ns["hartree_to_eV"](1.0), 27.2114, 1e-3),
             "<code>hartree_to_eV(1)</code> should be 27.211 eV. Multiply the energy by <code>HARTREE_TO_EV</code>."),
            (lambda ns: _close(ns["hartree_to_eV"](-0.5), -13.6057, 1e-3),
             "<code>hartree_to_eV(-0.5)</code> should be −13.606 eV. Make sure you <b>multiply</b> (not divide) and keep the sign."),
            (lambda ns: _close(ns["bohr_to_angstrom"](2.0), 1.05835, 1e-4),
             "<code>bohr_to_angstrom(2)</code> should be 1.0584 Å. One bohr is 0.529177 Å: multiply by <code>BOHR_TO_ANG</code>."),
        ],
        "ok": "The H atom's −0.5 hartree is −13.606 eV: exactly the ionization energy of hydrogen.",
    },
    "2.1": {
        "title": "H atom in several basis sets",
        "needs": ["h_atom_energy", "E_H"],
        "tests": [
            (lambda ns: ns["h_atom_energy"]("sto-3g") is not None,
             "<code>h_atom_energy</code> returns nothing. Its last line should be <code>return mf.kernel()</code>."),
            (lambda ns: _close(ns["h_atom_energy"]("sto-3g"), -0.466582, 1e-5),
             "<code>h_atom_energy('sto-3g')</code> should be −0.466582 hartree, as in the worked example. "
             "Did you use <code>scf.UHF(mol)</code> and return <code>mf.kernel()</code>?"),
            (lambda ns: all(b in ns["E_H"] for b in ["sto-3g", "6-31g", "cc-pvdz", "cc-pvtz", "cc-pvqz"]),
             "The dictionary <code>E_H</code> is missing some basis sets. Run the whole cell, including the loop."),
            (lambda ns: _close(ns["E_H"]["cc-pvqz"], -0.499946, 1e-5),
             "<code>E_H['cc-pvqz']</code> should be −0.499946 hartree. Rerun the cell after fixing the function."),
        ],
        "ok": "Every energy is above −0.5 hartree and falls as the basis grows: the variational principle at work.",
    },
    "2.2d": {"title": "Discussion: the variational principle", "open": "Question 2.2 Discussion", "min_chars": 60},
    "3.1": {
        "title": "H₂⁺ energy function",
        "needs": ["h2plus_energy"],
        "tests": [
            (lambda ns: ns["h2plus_energy"](1.057) > -0.7,
             "The energy is far too low. H₂⁺ has <b>one</b> electron: <code>charge = 1</code> (two protons, one electron)."),
            (lambda ns: ns["h2plus_energy"](1.057) < -0.55,
             "The energy is too high for H₂⁺. Check <code>charge = 1</code> and <code>spin = 1</code> "
             "(one electron, so one unpaired electron)."),
            (lambda ns: _close(ns["h2plus_energy"](1.057), -0.602244, 2e-5),
             "Close, but <code>h2plus_energy(1.057)</code> should be −0.602244 hartree with cc-pVTZ. "
             "Did you change the default basis set?"),
        ],
        "ok": "One electron, two protons, and a real bond: within 0.0004 hartree (11 meV) of the exact answer.",
    },
    "3.2": {
        "title": "H₂⁺ potential energy curve",
        "needs": ["E_h2p", "R_grid"],
        "tests": [
            (lambda ns: len(ns["E_h2p"]) == len(ns["R_grid"]),
             "<code>E_h2p</code> needs one energy for every R in <code>R_grid</code>. Use "
             "<code>[h2plus_energy(R) for R in R_grid]</code> inside <code>np.array(...)</code>."),
            (lambda ns: _close(ns["R_grid"][np.argmin(ns["E_h2p"])], 1.06, 0.1),
             "The lowest energy should be near R = 1.06 Å. Check that you call <code>h2plus_energy(R)</code> with "
             "<code>R</code> from the loop, not a fixed number."),
        ],
        "ok": "A bound molecule: the curve dips below H + H⁺ (−0.5 hartree) by 2.8 eV.",
    },
    "3.3": {
        "title": "H₂⁺ bond energy",
        "needs": ["De_h2p"],
        "tests": [
            (lambda ns: ns["De_h2p"] > 0,
             "<code>De_h2p</code> is negative. The bond energy is E(products) − E(molecule): "
             "<code>E_H['cc-pvtz'] - E_min_h2p</code>."),
            (lambda ns: ns["De_h2p"] > 0.5,
             "<code>De_h2p</code> looks like it is still in hartree. Convert it with <code>hartree_to_eV(...)</code>."),
            (lambda ns: 2.70 < ns["De_h2p"] < 2.85,
             "<code>De_h2p</code> should be about 2.79 eV. Use the H atom energy in the same basis, "
             "<code>E_H['cc-pvtz']</code>, and remember E(H⁺) = 0."),
        ],
        "ok": "2.79 eV against the exact 2.793 eV: the only approximation left is the basis set.",
    },
    "3.4d": {"title": "Discussion: where the bond comes from", "open": "Question 3.4 Discussion", "min_chars": 60},
    "4.1": {
        "title": "H₂ energy: Hartree–Fock and full CI",
        "needs": ["h2_energy"],
        "tests": [
            (lambda ns: _close(ns["h2_energy"](0.7414, "hf"), -1.132955, 2e-5),
             "<code>h2_energy(0.7414, 'hf')</code> should be −1.132955 hartree. The Hartree–Fock part was given: "
             "make sure it is unchanged."),
            (lambda ns: np.ndim(ns["h2_energy"](0.7414, "fci")) == 0,
             "The FCI energy should be a single number. <code>fci.FCI(mf).kernel()</code> returns (energy, wavefunction): "
             "add <code>[0]</code> to keep only the energy."),
            (lambda ns: _close(ns["h2_energy"](0.7414, "fci"), -1.172336, 2e-5),
             "<code>h2_energy(0.7414, 'fci')</code> should be −1.172336 hartree: "
             "<code>E_fci = fci.FCI(mf).kernel()[0]</code>."),
        ],
        "ok": "Full CI lies 1.07 eV below Hartree–Fock: that is the correlation energy of H₂.",
    },
    "4.2": {
        "title": "H₂ curves and correlation energy",
        "needs": ["E_hf_curve", "E_fci_curve", "E_corr", "R_grid2"],
        "tests": [
            (lambda ns: len(ns["E_hf_curve"]) == len(ns["R_grid2"]) and len(ns["E_fci_curve"]) == len(ns["R_grid2"]),
             "Both curves need one energy per R in <code>R_grid2</code> (not <code>R_grid</code>)."),
            (lambda ns: np.all(np.asarray(ns["E_fci_curve"]) <= np.asarray(ns["E_hf_curve"]) + 1e-8),
             "The full CI energy should never be above Hartree–Fock. Did you swap <code>'hf'</code> and <code>'fci'</code>?"),
            (lambda ns: np.all(np.asarray(ns["E_corr"]) < 1e-8),
             "The correlation energy is E(FCI) − E(HF), which is negative. Check the order of the subtraction."),
            (lambda ns: np.allclose(ns["E_corr"], np.asarray(ns["E_fci_curve"]) - np.asarray(ns["E_hf_curve"])),
             "<code>E_corr</code> should be <code>E_fci_curve - E_hf_curve</code>, point by point."),
        ],
        "ok": "Full CI breaks the bond into two H atoms; Hartree–Fock can't, and its error grows as the bond stretches.",
    },
    "4.3": {
        "title": "Harmonic wavenumber",
        "needs": ["harmonic_wavenumber"],
        "tests": [
            (lambda ns: ns["harmonic_wavenumber"](1.18424, 1.007825, 34.968853) > 0,
             "The wavenumber should be positive. Check the square root and the units."),
            (lambda ns: _close(ns["harmonic_wavenumber"](4 * 1.18424, 1.007825, 34.968853)
                               / ns["harmonic_wavenumber"](1.18424, 1.007825, 34.968853), 2.0, 1e-3),
             "A bond 4× stiffer should vibrate 2× faster (ω = √(k/μ)). Make sure you take the square root of k/μ."),
            (lambda ns: _close(ns["harmonic_wavenumber"](1.18424, 1.007825, 34.968853), 2991, 3),
             "The HCl check should give 2991 cm⁻¹. Check the conversions: k × <code>HARTREE_TO_J</code> / 1e-20 "
             "for N/m, μ × <code>AMU_TO_KG</code> for kg, and divide ω by 2π<i>c</i> with <i>c</i> in cm/s."),
        ],
        "ok": "Your function reproduces HCl's 2991 cm⁻¹; you'll reuse it for H₂ and O₂.",
    },
    "4.4": {
        "title": "H₂: bond energy and frequency",
        "needs": ["De_hf", "De_fci", "nu_hf", "nu_fci"],
        "tests": [
            (lambda ns: ns["De_fci"] > 0.5 and ns["De_hf"] > 0.5,
             "The bond energies should be positive and in eV: <code>hartree_to_eV(2 * E_H['cc-pvtz'] - E_min_fci)</code>."),
            (lambda ns: 4.6 < ns["De_fci"] < 4.8,
             "<code>De_fci</code> should be about 4.70 eV. Use <b>two</b> H atoms: <code>2 * E_H['cc-pvtz']</code>."),
            (lambda ns: 3.2 < ns["De_hf"] < 4.0,
             "<code>De_hf</code> should be about 3.6 eV. Use <code>E_min_hf</code> (not <code>E_min_fci</code>)."),
            (lambda ns: 4300 < ns["nu_fci"] < 4500 and ns["nu_hf"] > ns["nu_fci"],
             "The wavenumbers should be about 4590 (HF) and 4410 (FCI) cm⁻¹: <code>harmonic_wavenumber(k_fci, M_H, M_H)</code>."),
        ],
        "ok": "Full CI: 4.70 eV and 4409 cm⁻¹, against the exact 4.75 eV and 4401 cm⁻¹. Correlation matters most for D_e.",
    },
    "4.5d": {"title": "Discussion: why Hartree–Fock fails", "open": "Question 4.5 Discussion", "min_chars": 60},
    "5.1": {
        "title": "Counting determinants",
        "needs": ["n_det_H2", "n_det_O2"],
        "tests": [
            (lambda ns: ns["n_det_H2"] == 784,
             "H₂ has one α and one β electron in 28 orbitals: <code>math.comb(28, 1) * math.comb(28, 1)</code>."),
            (lambda ns: ns["n_det_O2"] == 8178045876000,
             "O₂ (triplet) has 9 α and 7 β electrons in 28 orbitals: <code>math.comb(28, 9) * math.comb(28, 7)</code>."),
        ],
        "ok": "8 trillion determinants for O₂ against 784 for H₂, with the same 28 orbitals: that's why we need DFT.",
    },
    "5.2": {
        "title": "O₂ energy with DFT",
        "needs": ["o2_energy"],
        "tests": [
            (lambda ns: _close(ns["o2_energy"](1.21, spin=2), -150.204643, 1e-4),
             "<code>o2_energy(1.21, spin=2)</code> should be −150.204643 hartree. Use <code>dft.UKS(mol)</code> when "
             "<code>spin > 0</code> and return <code>mf.kernel()</code>."),
            (lambda ns: ns["o2_energy"](1.21, spin=0) > ns["o2_energy"](1.21, spin=2),
             "For <code>spin=0</code> use <code>dft.RKS(mol)</code>: <code>dft.UKS(mol) if spin > 0 else dft.RKS(mol)</code>."),
        ],
        "ok": "The triplet is lower: O₂'s ground state has two unpaired electrons.",
    },
    "5.3": {
        "title": "O₂ triplet and singlet curves",
        "needs": ["E_trip", "E_sing", "R_grid3"],
        "tests": [
            (lambda ns: len(ns["E_trip"]) == len(ns["R_grid3"]) and len(ns["E_sing"]) == len(ns["R_grid3"]),
             "Both curves need one energy per R in <code>R_grid3</code>."),
            (lambda ns: np.all(np.asarray(ns["E_sing"]) > np.asarray(ns["E_trip"])),
             "The singlet should be above the triplet everywhere. Did you swap <code>spin=0</code> and <code>spin=2</code>?"),
            (lambda ns: 1.5 < (ns["E_sing"][np.argmin(ns["E_trip"])] - np.min(ns["E_trip"])) * 27.2114 < 1.9,
             "The singlet–triplet gap at the minimum should be about 1.7 eV. Check both curves used the default B3LYP/def2-SVP."),
        ],
        "ok": "A triplet ground state 1.7 eV below the closed-shell singlet: two unpaired electrons make O₂ paramagnetic.",
    },
    "5.4": {
        "title": "O₂ frequency and bond energy",
        "needs": ["nu_O2", "De_O2"],
        "tests": [
            (lambda ns: 1600 < ns["nu_O2"] < 1800,
             "<code>nu_O2</code> should be about 1700 cm⁻¹: <code>harmonic_wavenumber(k_O2, M_O, M_O)</code>."),
            (lambda ns: ns["De_O2"] > 0.5,
             "<code>De_O2</code> should be positive and in eV: <code>hartree_to_eV(2 * E_O - E_min_O2)</code>."),
            (lambda ns: 5.2 < ns["De_O2"] < 5.8,
             "<code>De_O2</code> should be about 5.5 eV. Use <b>two</b> O atoms: <code>2 * E_O</code>."),
        ],
        "ok": "B3LYP: 1697 cm⁻¹ and 5.5 eV, against 1580 cm⁻¹ and 5.21 eV measured. Good, but with no exact answer to check against.",
    },
    "5.5d": {"title": "Discussion: O₂ magnetism and IR", "open": "Question 5.5 Discussion", "min_chars": 60},
    "6d": {"title": "Reflection", "open": "**Question 6.**", "min_chars": 60},
}
