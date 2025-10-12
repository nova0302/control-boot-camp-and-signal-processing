import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE function: dy/dt = f(t, y)
def my_ode_function(t, y):
    # Example: y' = -2 * y
    return -2 * y

# Define the time span and initial condition
t_span = (0, 5)  # Integrate from t=0 to t=5
y0 = [1.0]       # Initial condition: y(0) = 1.0

# Solve the ODE
sol = solve_ivp(my_ode_function, t_span, y0, method='RK45', t_eval=np.linspace(0, 5, 100))

# Plot the solution
plt.plot(sol.t, sol.y[0], label='y(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of dy/dt = -2y')
plt.legend()
plt.grid(True)
plt.show()
