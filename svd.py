import numpy as np

# Define a sample matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Perform SVD
U, s, Vh = np.linalg.svd(A)

# Print the results
print("Original Matrix A:")
print(A)
print("\nLeft Singular Vectors (U):")
print(U)
print("\nSingular Values (s):")
print(s)
print("\nRight Singular Vectors (Vh):")
print(Vh)

# Reconstruct the original matrix (approximately)
# Create a diagonal matrix from singular values
S = np.zeros(A.shape)
S[:A.shape[1], :A.shape[1]] = np.diag(s)

# Reconstruct A
A_reconstructed = U @ S @ Vh
print("\nReconstructed Matrix A:")
print(A_reconstructed)
