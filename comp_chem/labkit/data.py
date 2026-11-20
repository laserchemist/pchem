"""Precomputed fallback data, so a slow server or a stuck student never blocks the next notebook."""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def data_path(name):
    """Absolute path of a file in the repository's data/ folder."""
    return os.path.join(os.path.dirname(_HERE), "data", name)


def load_co2_samples():
    """320 B3LYP/def2-SVP energies and dipoles of CO2 at displaced geometries.

    Returns a dict with
      symbols  : ['O', 'C', 'O']
      X        : (n, 3, 3) Cartesian geometries, angstrom
      dE_eV    : (n,) energy above the B3LYP/def2-SVP minimum, eV
      dipole_D : (n, 3) dipole vector, debye
      X0       : (3, 3) equilibrium geometry used for the sampling
    """
    d = np.load(data_path("co2_b3lyp_samples.npz"))
    return {"symbols": [str(s) for s in d["symbols"]], "X": d["X"], "dE_eV": d["dE_eV"],
            "dipole_D": d["dipole_D"], "X0": d["X0"], "method": str(d["method"])}


def results_path(name):
    """Where a lab saves a result for the next lab: comp_chem/results/<name> (the folder is created if needed)."""
    folder = os.path.join(os.path.dirname(_HERE), "results")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def find_result(name):
    """Path of a result saved by an earlier lab, or of the reference copy in data/ if there is none (says which)."""
    mine = os.path.join(os.path.dirname(_HERE), "results", name)
    if os.path.exists(mine):
        print(f"using your saved {name} (from the earlier lab)")
        return mine
    print(f"{name} not found in results/: using the reference copy in data/")
    return data_path(name)
