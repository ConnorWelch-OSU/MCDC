import matplotlib.pyplot as plt
import h5py
import numpy as np


# Load results
with h5py.File("output.h5", "r") as f:
    z = f["tally/grid/x"][:]
    dz = z[1:] - z[:-1]
    z_mid = 0.5 * (z[:-1] + z[1:])

    mu = f["tally/grid/g"][:]
    dmu = mu[1:] - mu[:-1]
    mu_mid = 0.5 * (mu[:-1] + mu[1:])

    psi = f["tally/flux/mean"][:]
    psi_sd = f["tally/flux/sdev"][:]

I = len(z) - 1
N = len(mu) - 1

# Scalar flux
phi = np.zeros(I)
phi_sd = np.zeros(I)

# Plotting Variables
E=np.linspace(1.0, 1e3, 1000)
bins=np.logspace(0.0, 3.0, 100)
width=np.diff(bins)

# Normalize
'''phi /= dz
phi_sd /= dz
for n in range(N):
    psi[:, n] = psi[:, n] / dz / dmu[n]
    psi_sd[:, n] = psi_sd[:, n] / dz / dmu[n]'''


# Flux - spatial average
'''plt.plot(mu_mid, psi, "-b", label="MC")
plt.xlabel(r"Energy, eV")
plt.ylabel("Flux")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.title(r"$\bar{\phi}_i$")
plt.show()'''

plt.step(mu_mid, psi / width, label="MC")
plt.plot(E, 1 / E, label="Analytic")
plt.xlabel(r"Energy, eV")
plt.ylabel("Flux")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

#print(mu_mid)
#print(psi)
