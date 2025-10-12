import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control as ct

m = 1
M = 5
L = 2
g = -10
d = 1

b = 1 # Pen up

A = np.array([[0,         1,                0,  0],
              [0,      -d/M,          b*m*g/M,  0],
              [0,         0,                0,  1],
              [0,-b*d/(M*L), -b*(m+M)*g/(M*L),  0]])

#B = np.array([[0], [1/M], [0], [b*1/(M*L)]]);
B = np.array([[0], [1/M], [0], [b*1/(M*L)]]);
ctrb_matrix = ct.ctrb(A,B)
#print(ctrb_matrix)
print(f'rank: {np.linalg.matrix_rank(ctrb_matrix)}')

evals, evecs= np.linalg.eig(A)
print(f'evals: {evals}')
eigs = np.array([-1.1, -1.2, -1.3, -1.4])

K = ct.place(A, B, eigs)
evals, evecs= np.linalg.eig(A-B*K)
print(f'evals: {evals}')
