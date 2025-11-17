import numpy as np
from scipy.integrate import solve_ivp

# Define system parameters (replace with actual values)
# These are placeholder values for the inverted pendulum on a cart.
# You will need to provide the actual physics constants.
m = 1  # mass of pendulum
M = 5  # mass of cart
L = 2  # length of pendulum
g = -10  # acceleration due to gravity
d = 1  # damping coefficient

# Define controller parameters (replace with actual values)
# The matrix 'K' represents the control gain, derived from a method like LQR.
# You must provide the correct K matrix for your system.
K = np.array([[10, 5, 20, 10]]) # Placeholder gain matrix for a 4-state system

def pendcart(t, x, m, M, L, g, d, u_func):
    """
    Defines the dynamics of the pendulum-cart system.
    
    Args:
        t: Time (scalar).
        x: State vector [cart_position, cart_velocity, pendulum_angle, pendulum_angular_velocity].
        m, M, L, g, d: Physical parameters of the system.
        u_func: A function handle for the control law.
        
    Returns:
        The derivative of the state vector, dx/dt.
    """
    
    # Unpack state variables
    x_pos, x_vel, theta, theta_vel = x
    
    # Calculate control input
    u = u_func(x)
    
    # Implement the full nonlinear dynamics of the cart-pendulum system.
    # The equations of motion can be complex and depend on the specific
    # model. The following is a general template.
    # Replace these with the actual, correct equations for your system.
    
    # Intermediate calculations for dynamics
    Sx = np.sin(theta)
    Cx = np.cos(theta)
    
    # These are placeholders; use the correct state-space representation.
    # A common form for the equations of motion for an inverted pendulum are:
    # d(theta_vel)/dt = ...
    # d(x_vel)/dt = ...
    # Replace the following lines with your specific equations.
    
    # Example dynamics (replace with correct physics equations)
    dx_vel = (u[0] + m * L * Sx * theta_vel**2 - m * g * Sx * Cx) / (M + m * Sx**2)
    dtheta_vel = (g * Sx * (M + m) - (u[0] * Cx) - (m * L * Cx * Sx * theta_vel**2)) / (L * (M + m * Sx**2))
    
    return [x_vel, dx_vel, theta_vel, dtheta_vel]


def simulate_closed_loop():
    """
    Simulates the closed-loop pendulum-cart system.
    """
    
    # Simulation settings
    tspan = (0, 10)  # Time span [start, end]
    t_eval = np.arange(tspan[0], tspan[1] + 0.001, 0.001)  # Time points for evaluation
    x0 = np.array([-1, 0, np.pi + 0.1, 0])  # Initial condition
    wr = np.array([1, 0, np.pi, 0])  # Reference position
    
    # Define the control law function
    u = lambda x: -K @ (x - wr)
    
    # Solve the system of ordinary differential equations
    # The 'args' parameter passes additional arguments to the `pendcart` function.
    sol = solve_ivp(
        lambda t, x: pendcart(t, x, m, M, L, g, d, u),
        tspan,
        x0,
        t_eval=t_eval,
        method='RK45' # RK45 is the default and equivalent to ode45
    )
    
    # The results are stored in the 'sol' object
    # t is the time vector, y is the solution array
    t = sol.t
    x = sol.y.T # Transpose to match MATLAB's output format (rows are states)
    
    # You can now plot or analyze the results
    print("Simulation complete. Results stored in 't' and 'x'.")

# Run the simulation
if __name__ == "__main__":
    print("Simulation complete. Results stored in 't' and 'x'.")
    simulate_closed_loop()
