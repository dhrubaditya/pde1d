import numpy as np
import matplotlib.pyplot as plt

# Read Fourier-space data
data = np.loadtxt("data/initcond_fourier.dat")
k = data[:, 0]
re = data[:, 1]
im = data[:, 2]
amp = np.sqrt(re**2 + im**2)

plt.figure(figsize=(6,4))
plt.plot(k, amp, label='|ψ̃(k)|')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.title('Initial Condition (Fourier Space)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

