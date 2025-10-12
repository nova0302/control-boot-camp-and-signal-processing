import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def pendcart(t, x, m, M, L, g, d, u):  # Added 't' as the first argument
    Sx = np.sin(x[2])  # Use np.sin
    Cx = np.cos(x[2])  # Use np.cos
    D  = m * L**2  * (M + m * (1 - Cx**2)) # Corrected power operator

    dx = np.zeros_like(x) # Initialize dx as a NumPy array

    dx[0] = x[1]
    dx[1] = (1/D) * (-m**2 * L**2 * g * Cx * Sx + m * L**2 * (m*L*x[3]**2 * Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D) * ((m+M) * m * g * L * Sx - m*L*Cx*(m*L*x[3]**2 * Sx - d*x[1])) - m*L*Cx*(1/D)*u
    return dx

m = 1
M = 5
L = 2
g = -10 # Corrected assignment
d = 1   # Corrected assignment
tMax = 50
tspan = (0,tMax)
x0 = [0, 0, np.pi, .5]

# Solve the ODE
# Pass parameters as a tuple in the 'args' argument
sol = solve_ivp(pendcart,
                tspan,
                x0,
                method='RK45',
                t_eval=np.linspace(0, tMax, 100),
                args=(m,M,L,g,d,0))
if s == -1:
    y0 = [[0],[0],[0],[0]]
    t,y = ode45


# Plot the solution
plt.plot(sol.t, sol.y[3], label='Cart Position (x)') # More descriptive label
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Cart-Pendulum System: Cart Position over Time')
plt.legend()
plt.grid(True)
plt.show()

'''


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def pendcart(x,m,M,L,g,d,u):
    Sx = sin(x[2])
    Cx = cos(x[2])
    D  = m * L * L * (M + m * (1-Cx^2))
    dx[0,0] = x[1]
    dx[1,0] = (1/D) * (-m^2 * L^2 * g * Cx * Sx + m * L^2 * (m*L*x[3]^2 * Sx - d*x[1])) + m*L*L*(1/D)*u
    dx[2,0] = x[3]
    dx[3,0] = (1/D) * ((m+M) * m * g * L * Sx - m*L*Cx*(m*L*x[3]^2 * Sx - d*x[1])) - m*L*Cx*(1/D)*u
    return dx

    
m = 1
M = 5
L = 2
g = -10
d = 1
tspan = (0,5)
x0 = [0,0,np.pi,.5]

# Solve the ODE
sol = solve_ivp(pendcart, tspan, x0, method='RK45', t_eval=np.linspace(0, 5, 100), args=(m,M,L,g,d,0))


# Plot the solution
plt.plot(sol.t, sol.y[0], label='y(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of dy/dt = -2y')
plt.legend()
plt.grid(True)
plt.show()






'''
