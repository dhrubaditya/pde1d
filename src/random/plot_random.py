import numpy as np
import matplotlib.pyplot as plt

# Read numeric data, skipping comment lines starting with '#'
data = np.loadtxt("random_numbers.txt", comments="#", dtype=np.float64)

plt.figure(figsize=(8, 5))
plt.hist(data, bins=50, color='skyblue', edgecolor='black', density=True)

plt.title("Histogram of Random Numbers")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

