"""Physical constants and unit conversions used throughout the lab.

Working units:
  * electronic structure (PySCF): hartree, bohr
  * everything else: eV, angstrom, amu, femtoseconds, cm^-1
"""

HARTREE_TO_EV = 27.211386245988      # 1 Eh in eV
BOHR_TO_ANG = 0.529177210903         # 1 a0 in angstrom
EV_TO_CM1 = 8065.543937              # 1 eV in cm^-1
HARTREE_TO_CM1 = 219474.6313632      # 1 Eh in cm^-1
KB_EV = 8.617333262e-5               # Boltzmann constant, eV/K
C_CM_PER_FS = 2.99792458e-5          # speed of light, cm/fs
DEBYE_PER_E_ANG = 4.80320471         # 1 e*angstrom in debye
HBAR_EV_FS = 0.6582119569            # reduced Planck constant, eV*fs

# Newton's law in lab units: a [A/fs^2] = F [eV/A] / m [amu] * ACC
ACC = 9.648533212e-3

# omega [rad/fs] from sqrt(k / m) with k in eV/A^2 and m in amu
#   omega = sqrt(k/m * ACC);  wavenumber [cm^-1] = omega / (2 pi c)
def omega_to_wavenumber(omega_rad_per_fs):
    """Convert an angular frequency in rad/fs to a wavenumber in cm^-1."""
    import numpy as np
    return np.asarray(omega_rad_per_fs) / (2 * np.pi * C_CM_PER_FS)

MASS = {  # amu, most abundant isotope
    "H": 1.007825, "D": 2.014102, "C": 12.000000, "N": 14.003074, "O": 15.994915,
    "F": 18.998403, "S": 31.972071, "Cl": 34.968853,
}

__all__ = ["HARTREE_TO_EV", "BOHR_TO_ANG", "EV_TO_CM1", "HARTREE_TO_CM1", "KB_EV",
           "C_CM_PER_FS", "DEBYE_PER_E_ANG", "HBAR_EV_FS", "ACC", "omega_to_wavenumber", "MASS"]
