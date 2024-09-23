import numpy as np
import os, h5py
from mpi4py import MPI

import sys
sys.path.append('C:/Users/conno/MCDC')
import mcdc


# Set the XS library directory
os.environ["MCDC_XSLIB"] = os.getcwd()

# =============================================================================
# Set source particles and create the source particle file
# (only master, rank 0)
# =============================================================================

if MPI.COMM_WORLD.Get_rank() == 0:
    rng = np.random.default_rng(seed=7)

    N = 1000
    bank = mcdc.make_particle_bank(N)

    for i in range(N):
        particle = bank[i]
        particle["x"] = rng.random() * 2.0
        particle["y"] = rng.random()
        particle["z"] = rng.random()
        particle["t"] = rng.random() * 2.0
        particle["ux"] = 1.0  # All going right
        particle["uy"] = 0.0
        particle["uz"] = 0.0
        particle["E"] = 20e6  # Arbitrary

    mcdc.save_particle_bank(bank, "source_particles")
MPI.COMM_WORLD.Barrier()


# =============================================================================
# Set model
# =============================================================================

# Create the dummy nuclide (only master, rank 0)
if MPI.COMM_WORLD.Get_rank() == 0:
    with h5py.File("dummy_nuclide.h5", "w") as f:
        f["A"] = 1.0

        f["E_xs"] = np.array([0.0, 1.0 - 1e-6, 1.0 + 1e-6, 2e7])
        f["capture"] = np.array([0.0, 0.0, 0.0, 0.0])
        f["fission"] = np.array([0.0, 0.0, 0.0, 0.0])
        f["scatter"] = np.array([1.0, 1.0, 1.0, 1.0])

        f["E_nu_p"] = np.array([0.0, 0.0, 0.0, 0.0])
        f["nu_p"] = np.array([0.0, 0.0, 0.0, 0.0])

        f["E_chi_p"] = np.array([0.0, 0.0, 0.0])
        f["chi_p"] = np.array([0.0, 0.0, 0.0])

        f["decay_rate"] = np.zeros(6)

        f["E_nu_d"] = np.array([0.0, 2e7])
        f["nu_d"] = np.zeros((6, 2))

        f["E_chi_d1"] = np.zeros(0)
        f["E_chi_d2"] = np.zeros(0)
        f["E_chi_d3"] = np.zeros(0)
        f["E_chi_d4"] = np.zeros(0)
        f["E_chi_d5"] = np.zeros(0)
        f["E_chi_d6"] = np.zeros(0)
        f["chi_d1"] = np.zeros(0)
        f["chi_d2"] = np.zeros(0)
        f["chi_d3"] = np.zeros(0)
        f["chi_d4"] = np.zeros(0)
        f["chi_d5"] = np.zeros(0)
        f["chi_d6"] = np.zeros(0)
MPI.COMM_WORLD.Barrier()

# Create the material
dummy_material = mcdc.material(
    [
        ["dummy_nuclide", 1.0],
    ]
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
s2 = mcdc.surface("plane-x", x=2.0, bc="reflective")

# Set cells
mcdc.cell(+s1 & -s2, dummy_material)


# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
mcdc.tally(
    scores=["flux"],
    E=np.logspace(0.0, 6.0, 100)
)

# Setting
mcdc.setting(source_file="source_particles.h5")

# Run
mcdc.run()
