import numpy as np
import matplotlib.pyplot as plt

# Load data from file (assumes whitespace-separated columns)
# Each line: x  f1(x)  f2(x)
data = np.loadtxt("data.txt")

# Extract columns
x = data[:, 0]
f1 = data[:, 1]
f2 = data[:, 2]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x, f1, 'ro-', label='f')  # red circles
plt.plot(x, f2, 'bs', label='f^3')  # blue squares
plt.plot(x, f1**3, 'k-', label='f^3 exact')  # red circles

# Labels and title
plt.xlabel('x')
plt.ylabel('Function ')
plt.title('cube')
plt.legend()
plt.grid(True)

# Show or save
plt.tight_layout()
plt.show()

