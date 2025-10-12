import control as ct
import numpy as np

# Define state-space matrices (example values)
A = np.array([[0, 1], [-4, -5]])
B = np.array([[0], [1]])

# Define desired closed-loop pole locations
p = np.array([-1, -2])

# Calculate the gain matrix K using the place function
K = ct.place(A, B, p)

print("Gain matrix K:")
print(np.linalg.eig(A-B*K))

import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Transpose the array using .T
transposed_arr = arr.T
print(transposed_arr)

C = np.array([1, 2, 4])
print(C)
C = C.T
print(np.transpose(C))
