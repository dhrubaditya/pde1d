import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Read optional command-line arguments for Q values
Q1 = int(sys.argv[1]) if len(sys.argv) > 1 else 1
Q2 = int(sys.argv[2]) if len(sys.argv) > 2 else None

#datasets = [Q1] + ([Q2] if Q2 is not None else [])

plt.figure(figsize=(6, 4))
any_plotted = False

if Q2 is not None:
    datasets = list(range(Q1, Q2 + 1))
else:
    datasets = [Q1]
i=0
for Q in datasets:
    filename = f"data/spec{Q}.dat"
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        continue

    data = np.loadtxt(filename)
    k = data[:, 0]
    spectrum = data[:, 1]
    plt.loglog(k[1:], spectrum[1:], label=f'Spectrum Q={Q}')
    any_plotted = True

if not any_plotted:
    print("No data files found. Exiting.")
    exit(1)

plt.xlabel('k')
plt.ylabel('Spectrum(k)')
plt.title('Spectrum Plot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
