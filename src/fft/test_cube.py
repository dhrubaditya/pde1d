def nonlin(a, b):
    """
    Takes two real arrays a and b, forms psi = a + 1j*b,
    and returns abs(psi**2) * psi.
    """
    psi = a + 1j * b
    result = (psi * np.conj(psi) )  * psi
    return result

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Load data from file (assumes whitespace-separated columns)
# Each line: x  f1(x)  f2(x)
data = np.loadtxt("data.txt")

# Extract columns
x = data[:, 0]
f_re = data[:, 1]
f_im = data[:, 2]
df_re = data[:, 3]
df_im = data[:, 4]

nlin = nonlin(f_re, f_im);
# Create the plot
plt.figure(1)
plt.plot(x, f_re,  label='psi')  
plt.plot(x, df_re,  label='|psi^2|psi_code')  
plt.plot(x, np.real(nlin),  label='|psi^2|psi_here')  
# Labels and title
plt.xlabel('x')
plt.ylabel('Function ')
plt.title('real part')
plt.legend()
plt.grid(True)

plt.figure(2)
plt.plot(x, f_im,  label='f.im')  
plt.plot(x, df_im,  label='df.im')  
plt.plot(x, np.imag(nlin),  label='df.im')  
# Labels and title
plt.xlabel('x')
plt.ylabel('Function ')
plt.title(' imag part')
plt.legend()
plt.grid(True)


# Show or save
plt.tight_layout()
plt.show()

