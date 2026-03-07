import numpy as np
import sys as sys
import matplotlib.pyplot as plt

default = "initcond"
# Use the first argument after the script name if provided
value = sys.argv[1] if len(sys.argv) > 1 else default
fname = "data/"+value+"_real.dat"
# Read data
data = np.loadtxt(fname)
x = data[:, 0]
psi = data[:, 1]

plt.figure(figsize=(6,4))
plt.plot(x, psi, label='psi(x)')
plt.xlabel('x')
plt.ylabel('psi(x)')
plt.title('Real Space')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

