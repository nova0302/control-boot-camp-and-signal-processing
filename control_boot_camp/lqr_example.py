import numpy as np
import control as ct

# Define system matrices (e.g., for a mass-spring-damper system)
A = np.array([[0, 1], [-1, -0.1]])
B = np.array([[0], [1]])

# Define state and input weighting matrices
Q = np.array([[10, 0], [0, 1]])  # Penalize state deviations
R = np.array([[0.1]])          # Penalize control effort

# Design the LQR controller
K, S, E = ct.lqr(A, B, Q, R)

print("Optimal gain matrix K:\n", K)
# K is the feedback gain matrix such that u = -Kx

# Further steps would involve simulating the system with the LQR controller
# and analyzing its performance (e.g., step response).
