import numpy as np
import matplotlib.pyplot as plt

# Load data
# Assumes two columns: k  |  |F(k)|^2
data = np.loadtxt("spectrum.txt")

k = data[:, 0]          # mode index
spectrum = data[:, 1]   # |F(k)|^2

# Plot
plt.figure(figsize=(8,5))
plt.plot(k, spectrum, marker='o', linestyle='-', color='b')
plt.xlabel("Mode k")
plt.ylabel("|F(k)|^2")
plt.title("Power Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()

