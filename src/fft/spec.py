import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=int, default=1,
                    help=" loglog plot if set to default (1)  ")
args = parser.parse_args()

# Load data
# Assumes two columns: k  |  |F(k)|^2
data = np.loadtxt("spectrum.txt")

k = data[:, 0]          # mode index
spectrum = data[:, 1]   # |F(k)|^2

# Plot
plt.figure(figsize=(8,5))
if args.log != 1 :
    plt.plot(k, spectrum, marker='.', linestyle='-', color='b')
else:
    plt.loglog(k, spectrum, marker='.', linestyle='-', color='b')
plt.xlabel("Mode k")
plt.ylabel("|F(k)|^2")
plt.title("Power Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()

