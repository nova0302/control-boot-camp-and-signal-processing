import control as ct
import numpy as np

# Define system matrices
A = np.array([[0, 1], [-1, -0.5]])
G = np.array([[0], [1]])
C = np.array([[1, 0]])

# Define noise covariances
QN = np.array([[0.1]])  # Process noise covariance
RN = np.array([[0.01]]) # Sensor noise covariance

# Compute the LQE gain
L, P, E = ct.lqe(A, G, C, QN, RN)

print("Observer Gain (L):\n", L)
print("Error Covariance (P):\n", P)
print("Estimator Eigenvalues (E):\n", E)
