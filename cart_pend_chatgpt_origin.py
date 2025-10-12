import numpy as np
from scipy.integrate import solve_ivp

# Define the control law as a Python function (equivalent to the MATLAB function handle).
# In MATLAB: u=@(x)-K*(x - wr);
# This assumes 'wr' is a target state vector and 'K' is a gain matrix.
def control_law(x, K, wr):
    """
    Calculates the control input u based on a state feedback law.

    Args:
        x (ndarray): The current state vector.
        K (ndarray): The feedback gain matrix.
        wr (ndarray): The reference or target state vector.

    Returns:
        float: The control input u.
    """
    # Matrix multiplication, similar to MATLAB's K*(x - wr)
    return -np.dot(K, (x - wr))

def pendcart(t, x, m, M, L, g, d, K, wr):
    """
    Defines the system dynamics for the cart-pendulum.

    This function represents the right-hand side of the differential equations,
    with the control input 'u' embedded.

    Args:
        t (float): Current time (required by solve_ivp).
        x (ndarray): The current state vector [position, velocity, angle, angular_velocity].
        m (float): Mass of the pendulum bob.
        M (float): Mass of the cart.
        L (float): Length of the pendulum rod.
        g (float): Acceleration due to gravity.
        d (float): Damping coefficient.
        K (ndarray): Feedback gain matrix for the controller.
        wr (ndarray): Reference state vector for the controller.

    Returns:
        ndarray: The derivative of the state vector, [x_dot, x_ddot, theta_dot, theta_ddot].
    """
    # Extract states from the state vector x
    pos, vel, theta, theta_dot = x

    # Calculate the control input u using the control law
    u = control_law(x, K, wr)

    # Simplified constants for clarity
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    D = m * L**2 * (M + m * (1 - cos_theta**2))

    # Calculate derivatives (state-space representation)
    dxdt = np.zeros(4)
    dxdt[0] = vel
    dxdt[1] = (1 / D) * (-m**2 * L**2 * g * cos_theta * sin_theta +
                         m * L**2 * (m * L * theta_dot**2 * sin_theta - d * vel)) + u * m * L**2 / D
    dxdt[2] = theta_dot
    dxdt[3] = (1 / D) * ((m + M) * m * g * L * sin_theta -
                         m * L * cos_theta * (m * L * theta_dot**2 * sin_theta - d * vel)) - u * m * L * cos_theta / D
    return dxdt

# --- Main simulation script ---

if __name__ == '__main__':
    # 1. Define model parameters
    m = 1.0  # Pendulum mass
    M = 5.0  # Cart mass
    L = 2.0  # Pendulum length
    g = 9.81 # Gravity
    d = 1.0  # Damping coefficient

    # 2. Define control law parameters
    # Note: K and wr would typically be designed using control theory methods.
    # For this example, these are placeholder values.
    K = np.array([[-10, -5, 50, 10]]) # Example gain matrix
    wr = np.array([0, 0, 0, 0]) # Example reference state (e.g., upright position at origin)

    # 3. Set up the simulation
    tspan = [0, 20] # Time span [start, end]
    x0 = np.array([0, 0, np.pi + 0.1, 0]) # Initial state: [position, velocity, angle, angular_velocity]
    # The initial angle is set slightly off vertical (np.pi) to observe the controller.

    # 4. Integrate the differential equations using `solve_ivp`
    # We pass the system parameters `m`, `M`, `L`, etc., via the `args` parameter.
    solution = solve_ivp(
        lambda t, x: pendcart(t, x, m, M, L, g, d, K, wr),
        tspan,
        x0,
        dense_output=True
    )

    # 5. Extract and display the results
    t = solution.t
    x = solution.y

    print("Simulation finished.")
    print(f"Number of time steps: {len(t)}")
    print("Final state:")
    print(f"  Position: {x[0, -1]:.4f}")
    print(f"  Velocity: {x[1, -1]:.4f}")
    print(f"  Angle:    {x[2, -1]:.4f} rad")
    print(f"  Ang Vel:  {x[3, -1]:.4f} rad/s")
    
    # Example plotting using matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, x[0, :], label='Cart Position (x)')
        plt.plot(t, x[2, :], label='Pendulum Angle (theta)')
        plt.title('Cart-Pendulum Simulation with Control')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not found. Skipping plotting.")
