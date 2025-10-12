import control as ct
import numpy as np

# Example: Define a state-space system
# dx/dt = Ax + Bu
# y = Cx + Du
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])

# Calculate the controllability matrix
C_matrix = ct.ctrb(A, B)

print("Controllability Matrix:")
print(C_matrix)

# Check the rank to determine controllability
rank_C = np.linalg.matrix_rank(C_matrix)
print(f"Rank of Controllability Matrix: {rank_C}")

# For a system with n states, if rank(C_matrix) == n, the system is controllable.
if rank_C == A.shape[0]:
    print("The system is controllable.")
else:
    print("The system is not controllable.")
