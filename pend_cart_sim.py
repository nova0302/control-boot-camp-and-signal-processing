import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ======================================================
# 1. System Parameters (Assumed values for cart-pendulum)
# ======================================================
m = 1   # Mass of the pendulum bob (kg)
M = 5.0   # Mass of the cart (kg)
L = 2.0   # Length of the pendulum rod (m)
g = -10  # Gravity (m/s^2)
d = 1   # Friction coefficient (cart)

# ======================================================
# 2. Define Control Law (LQR Controller Placeholder K)
# ======================================================
# NOTE: You must replace this 'K' matrix with the actual gain matrix 
# derived from your LQR design for your specific system parameters. 
# This is just a placeholder matrix shape (1 input x 4 states).
#K = np.array([[-10.0, -5.0, 50.0, 10.0]]) 
K = np.array([-31.6227766, -73.58924546, 1004.80602597,  495.65703441]) 

# Reference state: [position, velocity, angle, angular_rate]
# Upright position is angle = pi (or 0, depending on definition. We use pi for inverted).
wr = np.array([0.0, 0.0, np.pi, 0.0]) 

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

# ======================================================
# 3. Define System Dynamics (pendcart equivalent)
# ======================================================
#def pendcart(t,x):
def pendcart(t,x,m,M,L,g,d,u):
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = m * L**2 * ( M + m * ( 1 - Cx**2 ))
    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[1] = (1/D)*(-m**2*L**2*g*Cx*Sx + m*L**2*(m*L*x[3]**2*Sx - d *x[1])) + m*L*L*(1/D)*u
    dx[2] = x[3]
    dx[3] = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*x[3]**2*Sx - d* x[1])) - m*L*Cx*(1/D)*u
    return dx
    
def pendcart_dynamics(t, x):
    """
    Nonlinear dynamics function for the cart-pendulum system: dx/dt = f(x, u)
    State vector x = [x (pos), x_dot (vel), theta (angle), theta_dot (ang. vel)]
    """
    # Unpack states
    x_cart, x_dot, theta, theta_dot = x
    
    # Calculate control input at the current state
    u = control_law(x)

    # Nonlinear Equations of Motion (derived assuming angle is vertical up=pi or 0)
    # Note: These equations are complex; ensure they match your MATLAB model.
    
    SinTheta = np.sin(theta)
    CosTheta = np.cos(theta)
    
    # Precompute denominators/complex terms
    Denom = M + m * SinTheta**2
    
    # Equations for accelerations (x_ddot and theta_ddot)
    x_ddot = (u + m * L * theta_dot**2 * SinTheta - d * x_dot) / Denom
    
    theta_ddot = (-u * CosTheta - m * L * theta_dot**2 * SinTheta * CosTheta 
                  - (M + m) * g * SinTheta - d * x_dot * CosTheta) / (L * Denom)
    
    # Return the state derivatives [x_dot, x_ddot, theta_dot, theta_ddot]
    return [x_dot, x_ddot, theta_dot, theta_ddot]

# ======================================================
# 4. Simulation Setup
# ======================================================
t_span_matlab = [0, 10]  # Start and end time
dt = 0.001
tspan = np.arange(t_span_matlab[0], t_span_matlab[1] + dt, dt) 

# Initial condition: [pos, vel, angle (near upright), ang_vel]
x0 = np.array([0, 0, np.pi + 0.1, 0]) # Matches MATLAB x0

print(f"Starting simulation from t={t_span_matlab[0]} to t={t_span_matlab[1]} seconds...")

# ======================================================
# 5. Run Simulation (using solve_ivp, equivalent to ode45)
# ======================================================
# We use dense_output=True and t_eval to get results at specified timesteps
#sol = solve_ivp(
#    fun=pendcart_dynamics,      # The dynamics function
#    t_span=t_span_matlab,       # [Start time, End time]
#    y0=x0,                      # Initial state
#    t_eval=tspan,               # Times at which to store the solution
#    method='RK45',              # Equivalent to MATLAB's ode45
#    rtol=1e-5,                  # Relative tolerance
#    atol=1e-5                   # Absolute tolerance
#)

sol = solve_ivp(
    fun=pendcart,      # The dynamics function
    t_span=t_span_matlab,       # [Start time, End time]
    y0=x0,                      # Initial state
    t_eval=tspan,               # Times at which to store the solution
    args=(m,M,L,g,d,0)
)

# Extract results
t = sol.t
x_sim = sol.y.T # Transpose to get columns as states: [time_steps, 4_states]

print("Simulation finished.")

# ======================================================
# 6. Plot Results
# ======================================================
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Closed-Loop Inverted Pendulum Simulation')

axs[0].plot(t, x_sim[:, 0])
axs[0].axhline(y=wr[0], color='r', linestyle='--', linewidth=0.8)
axs[0].set_ylabel('Cart Position (x)')
axs[0].grid(True)

axs[1].plot(t, x_sim[:, 1])
axs[1].set_ylabel('Cart Velocity ($\dot{x}$)')
axs[1].grid(True)

# Plot angle, subtracting pi to show deviation from upright
axs[2].plot(t, x_sim[:, 2] - np.pi) 
axs[2].set_ylabel('Pendulum Angle Dev. ($\Theta$)')
axs[2].grid(True)

axs[3].plot(t, x_sim[:, 3])
axs[3].set_ylabel('Angle Rate ($\dot{\Theta}$)')
axs[3].set_xlabel('Time (s)')
axs[3].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
