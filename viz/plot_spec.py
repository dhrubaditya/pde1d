import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Read optional command-line argument for Q
Q = int(sys.argv[1]) if len(sys.argv) > 1 else 1
filename = f"data/spec{Q}.dat"

if not os.path.exists(filename):
    print(f"Error: {filename} not found.")
    exit(1)

data = np.loadtxt(filename)
k = data[:, 0]
spectrum = data[:, 1]

plt.figure(figsize=(6,4))
plt.loglog(k, spectrum, '.-',label=f'Spectrum Q={Q}')
plt.xlabel('k')
plt.ylabel('Spectrum(k)')
plt.title(f'Spectrum Plot (spec{Q}.dat)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

