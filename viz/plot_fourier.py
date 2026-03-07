import numpy as np
import sys as sys
import os as os
import matplotlib.pyplot as plt

default = "initcond"
# Use the first argument after the script name if provided
value = sys.argv[1] if len(sys.argv) > 1 else default
fname = "data/"+value+"_fourier.dat"
#if (not os.path.isfile):
#	print(fname,"not found")
#	sys.exit(1)
# Read Fourier-space data
data = np.loadtxt(fname)
k = data[:, 0]
re = data[:, 1]
im = data[:, 2]
amp = np.sqrt(re**2 + im**2)

plt.figure(figsize=(6,4))
plt.plot(k, amp, label='|ψ̃(k)|')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.title('psi (Fourier Space)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

