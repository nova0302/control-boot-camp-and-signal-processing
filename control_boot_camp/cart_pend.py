import numpy as np
from scipy.integrate import solve_ivp

# Define the control law as a Python function (equivalent to the MATLAB function handle).
# In MATLAB: u=@(x)-K*(x - wr);
# This assumes 'wr' is a target state vector and 'K' is a gain matrix.

def control_law(x, K, wr):
    return -np.dot(K, (x - wr))

def pendcart(t, x, m, M, L, g, d, K, wr):
    # Extract states from the state vector x
    pos, vel, theta, theta_dot = x

    # Calculate the control input u using the control law
    u = control_law(x, K, wr)

    # Simplified constants for clarity
    Stheta = np.sin(theta)
    Ctheta = np.cos(theta)
    D = m * L**2 * (M + m * (1 - Ctheta**2))

    # Calculate derivatives (state-space representation)
    #dx = np.zeros(4)
    dx = np.zeros_like(x)
    dx[0] = vel
    dx[1] = (1 / D) * (-m**2 * L**2 * g * Ctheta * Stheta + m * L**2 * (m * L * theta_dot**2 * Stheta - d * vel)) + u * m * L**2 / D
    dx[2] = theta_dot
    dx[3] = (1 / D) * ((m + M) * m * g * L * Stheta - m * L * Ctheta * (m * L * theta_dot**2 * Stheta - d * vel)) - u * m * L * Ctheta / D
    return dx

# --- Main simulation script ---

if __name__ == '__main__':
    # 1. Define model parameters
    m = 1  # Pendulum mass
    M = 5  # Cart mass
    L = 2  # Pendulum length
    g = -10 # Gravity
    d = 1  # Damping coefficient
    
    # 2. Define control law parameters
    # Note: K and wr would typically be designed using control theory methods.
    # For this example, these are placeholder values.
    #K = np.array([[-10, -5, 50, 10]]) # Example gain matrix
    K = np.array([[-31.6227766,   -73.58924546, 1004.80602597,  495.65703441]]) # Example gain matrix

    wr = np.array([1, 0, np.pi, 0]) # Example reference state (e.g., upright position at origin)
    
    # 3. Set up the simulation
    tspan = [0, 20] # Time span [start, end]
    x0 = np.array([-1, 0, np.pi + 0.1, 0]) # Initial state: [position, velocity, angle, angular_velocity]
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
        plt.plot(t, x[1, :], label='Cart Velacity (x)')
        plt.plot(t, x[2, :], label='Pendulum Angle (theta)')
        plt.plot(t, x[3, :], label='Pendulum Anglar Speed (theta)')
        plt.title('Cart-Pendulum Simulation with Control')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not found. Skipping plotting.")
