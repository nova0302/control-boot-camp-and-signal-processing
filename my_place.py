import numpy as np
import control as ct

# Define your system matrices A and B
A = np.array([[0, 1], [-4, -5]])
B = np.array([[0], [1]])

# Define the desired closed-loop pole locations
p = np.array([-2, -3])

# Calculate the state feedback gain matrix K using pole placement
K = ct.place(A, B, p)

print("Gain matrix K:")
print(K)
