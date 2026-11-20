"""labkit: plumbing for the computational physical chemistry lab.

Everything that is *physics* (energies, Hessians, normal modes, the GP surface,
velocity Verlet, the IR spectrum) is written by the students in the notebooks.
This package only holds the parts that are not the point of the lab:
unit constants, 3-D viewers and animations, small plotting helpers and the
precomputed fallback data.
"""
from .units import *          # noqa: F401,F403
from .viz import show_molecule, animate_mode, animate_trajectory, trajectory_strip  # noqa: F401
from .data import load_co2_samples, data_path, results_path, find_result  # noqa: F401

__version__ = "1.0"
