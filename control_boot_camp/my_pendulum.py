import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control as ct

# Define the function for the system of first-order ODEs
def pendulum(t, y, g, L):
    theta, omega = y
    dydt = [omega, - (g / L) * np.sin(theta)]
    return dydt

# Parameters and initial conditions
g = -10
L = 1
t_span = (0, 20)
y0 = [np.pi / 2, 0]  # [initial angle, initial angular velocity]

# Solve the ODE
sol_pendulum = solve_ivp(
    fun=pendulum,
    t_span=t_span,
    y0=y0,
    t_eval=np.linspace(0, 20, 200),
    args=(g, L)
)

# Plot the results
plt.plot(sol_pendulum.t, sol_pendulum.y[0], label='Angle (rad)')
plt.plot(sol_pendulum.t, sol_pendulum.y[1], label='Angular Velocity (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('Pendulum Motion')
plt.legend()
plt.grid()
plt.show()
