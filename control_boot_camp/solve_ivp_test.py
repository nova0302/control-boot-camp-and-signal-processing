import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define the ODE system as a function.
# The function must take `t` and `y` as arguments, in that order.
def exponential_decay(t, y):
    return -0.5 * y

# 2. Define the time span and initial condition.
t_span = (0, 10) # Solve from t=0 to t=10
y0 = [1]         # Initial value y(0) = 1. Must be a list or array.

# 3. Choose the time points where the solution should be evaluated.
# If `t_eval` is not specified, the solver will select time points.
t_eval = np.linspace(0, 10, 100)

# 4. Call solve_ivp.
solution = solve_ivp(exponential_decay, t_span, y0, t_eval=t_eval)

# 5. The result is an object with solution data.
t_values = solution.t
y_values = solution.y[0] # The solution values for y

# 6. Plot the results.
plt.figure(figsize=(8, 6))
plt.plot(t_values, y_values, label='Numerical solution', marker='o', markersize=4)
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.title('Solution to dy/dt = -0.5y')
plt.legend()
plt.grid(True)
plt.show()
