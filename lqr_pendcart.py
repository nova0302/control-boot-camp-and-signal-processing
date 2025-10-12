import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control as ct

# Define the control law function u(x)
def control_law(x):
    """u = -K * (x - wr)"""
    # Calculate state deviation
    x_deviation = x - wr
    
    # Handle angle wrapping if necessary (optional, depending on simulation model)
    # if x_deviation[2] > np.pi: x_deviation[2] -= 2 * np.pi
    # if x_deviation[2] < -np.pi: x_deviation[2] += 2 * np.pi

    # Calculate control input
    u = -K @ x_deviation
    # Assuming u is a scalar input for a single cart force
    return u[0] 

def pendcart(t,x,m,M,L,g,d,K, wr):
    u = -np.dot(K, (x-wr))

    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m * L**2 * ( M + m * ( 1 - Cx**2 ))

    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[1] = (1/D)*(-m**2 * L**2 * g * Cx * Sx + m * L**2 * (m * L * x[3]**2 * Sx - d * x[1])) + m * L**2 * (1/D) * u
    dx[2] = x[3]
    dx[3] = (1/D) * ((m+M) * m * g * L * Sx - m * L * Cx *(m * L * x[3]**2 * Sx - d * x[1])) - m * L * Cx* (1/D) *u
    return dx

m = 1
M = 5
L = 2
g = -10
d = 1

b = 1 # Pen up
#b = -1 # Pen Down

A = np.array([[0,         1,                0,  0],
              [0,      -d/M,          b*m*g/M,  0],
              [0,         0,                0,  1],
              [0,-b*d/(M*L), -b*(m+M)*g/(M*L),  0]])

B = np.array([[0], [1/M], [0], [b/(M*L)]]);

T, D = np.linalg.eig(A)

# 1. unstable cause there is an eig val that has positive real part like 2.4647
print(f'T:{T.real}')
print("\n")
ctrb_matrix = ct.ctrb(A,B)
print(f'ctrb: {ctrb_matrix}')

# 2. ctrb because ctrb matrix has full rank(4)
print(f'rank: {np.linalg.matrix_rank(ctrb_matrix)}')

#3. A,B is ctrb, we can place eig vals anywhere we want
eigs = np.array([-1.1, -1.2, -1.3, -1.4])

K = ct.place(A,B,eigs)
T, D = np.linalg.eig(A-B*K)

#4. we see that eigs and T are the same
print(f'--T:{T.real}')
print("\n")

Q = np.array([[1, 0,  0,   0],
              [0, 1,  0,   0],
              [0, 0, 10,   0],
              [0, 0,  0, 100]])

R = 0.001

K, S, E = ct.lqr(A, B, Q, R)
print(f'K:{K}')
T, D = np.linalg.eig(A-B*K)


#5. now *all eig vals* have neg real eig vals, which means the system is stable
print(f'T:{T.real}')
print("\n")

print(np.diag(D.real))
print("\n")

d_r = D.real
print(d_r)
print("\n")

diag_d = np.diag(D)
print(diag_d)
print("\n")

diag_d_r = np.diag(d_r)
print(diag_d_r)
print("\n")

# ======================================================
# 4. Simulation Setup
# ======================================================
t_span_matlab = [0, 10]  # Start and end time
dt = 0.001
tspan = np.arange(t_span_matlab[0], t_span_matlab[1] + dt, dt) 

# Initial condition: [pos, vel, angle (near upright), ang_vel]
x0 = np.array([0, 0, np.pi + 0.1, 0]) # Matches MATLAB x0

print(f"Starting simulation from t={t_span_matlab[0]} to t={t_span_matlab[1]} seconds...")

sol = solve_ivp(
    fun=pendcart,      # The dynamics function
    t_span=t_span_matlab,       # [Start time, End time]
    y0=x0,                      # Initial state
    t_eval=tspan,               # Times at which to store the solution
    args=(m,M,L,g,d,K,wr))
