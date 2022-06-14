import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different materials

# Set materials
m1 = mcdc.material(capture=np.array([50.0]))
m2 = mcdc.material(capture=np.array([5.0]))
m3 = mcdc.material(capture=np.array([0.0])) # Vacuum
m4 = mcdc.material(capture=np.array([0.1]), scatter=np.array([[0.9]]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0.0, bc="reflective")
s2 = mcdc.surface('plane-x', x=2.0)
s3 = mcdc.surface('plane-x', x=3.0)
s4 = mcdc.surface('plane-x', x=5.0)
s5 = mcdc.surface('plane-x', x=8.0, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m1)
mcdc.cell([+s2, -s3], m2)
mcdc.cell([+s3, -s4], m3)
mcdc.cell([+s4, -s5], m4)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the medium

mcdc.source(x=[0.0, 2.0], isotropic=True, prob=100.0)
mcdc.source(x=[5.0, 6.0], isotropic=True, prob=1.0)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally(scores=['flux', 'current', 'flux-x', 'current-x'], 
           x=np.linspace(0.0, 8.0, 41))

# Setting
mcdc.setting(N_hist=1E5)

# Run
mcdc.run()
