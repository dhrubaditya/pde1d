import numpy as np
import matplotlib.pyplot as plt

# Read data
data = np.loadtxt("data/initcond_real.dat")
x = data[:, 0]
psi = data[:, 1]

plt.figure(figsize=(6,4))
plt.plot(x, psi, label='psi(x)')
plt.xlabel('x')
plt.ylabel('psi(x)')
plt.title('Initial Condition (Real Space)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

