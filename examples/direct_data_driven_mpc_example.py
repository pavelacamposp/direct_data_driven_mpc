"""
Direct Data-Driven Model Predictive Control (MPC) Example Script

This script demonstrates the setup, simulation, and visualization of a Direct 
Data-Driven MPC controller applied to a four-tank system model based on
research by J. Berberich et al., as described in [1]. The implementation uses
the parameters of the system model, the initial input-output data generation
and the Data-Driven MPC controller, as defined in the example presented in 
Section V of [1].

To illustrate a typical controller operation, this script does not set the
initial output state of the system to `y_0 = [0.4, 0.4]`, as shown in the
closed-loop output graphs from Fig. 2 in [1]. Instead, the initial system
state is estimated with a randomized input sequence.

Additionally, the input-output equilibrium pair `u_s = [[1], [1]]`,
`y_s = [[0.65], [0.77]]` defined in the example from the paper is not an exact
trajectory point of the four-tank system model, since the equilibrium output
corresponding to `u_s` is `y_s = [[0.6444], [0.7526]]`. This does not
represent an issue for the Robust Data-Driven MPC controller due to the
addition of the slack variable that relaxes the system dynamics constraint.
However, for the Nominal Data-Driven MPC, this difference leads to unfeasible
solutions.

To provide a unified script for illustrating the functionality of both Nominal
and Robust Data-Driven MPC controllers, we consider the input-output
equilibrium pair `u_s = [[1], [1]]`, `y_s = [[0.64440373], [0.75261324]]`,
calculating `y_s` from `u_s` using the system model matrices.

For a closer approximation of the results presented in the paper using only
Robust Data-Driven MPC controllers, considering the initial output state of
the system `y_0 = [0.4, 0.4]`, and the defined input-output equilibrium pair
`u_s = [[1], [1]]`, `y_s = [[0.65], [0.77]]`, as presented in the paper
example from [1], please refer to 'robust_data_driven_mpc_reproduction.py'.

References:
    [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Data-Driven
        Model Predictive Control With Stability and Robustness Guarantees," in
        IEEE Transactions on Automatic Control, vol. 66, no. 4, pp. 1702-1717,
        April 2021, doi: 10.1109/TAC.2020.3000182.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import os

from utilities.initial_state_estimation import (
    observability_matrix, toeplitz_input_output_matrix,
    estimate_initial_state, calculate_output_equilibrium_setpoint)
from utilities.data_visualization import (
    plot_input_output, plot_input_output_animation, save_animation)

from direct_data_driven_mpc.direct_data_driven_mpc_controller import (
    DirectDataDrivenMPCController, DataDrivenMPCType, SlackVarConstraintTypes)

from models.four_tank_system import FourTankSysParams

# Directory paths
dirname = os.path.dirname
project_directory = dirname(dirname(__file__))
default_animation_dir = os.path.join(project_directory, 'animation_outputs')

# Animation video default parameters
video_name = "data-driven_mpc_sim.mp4"
default_video_path = os.path.join(default_animation_dir, video_name)
default_video_fps = 100
default_video_bitrate = 1800

# Data-Driven MPC controller parameters
controller_type_mapping = {
    "Nominal": DataDrivenMPCType.NOMINAL,
    "Robust": DataDrivenMPCType.ROBUST,
}
default_eps_max = 0.002
default_t_sim = 600 # Default simulation length in time steps

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Data-Driven MPC "
                                     "Controller Example")
    # Data-Driven MPC controller arguments
    parser.add_argument("--n_mpc_step", type=int,
                        default=None,
                        help="The number of consecutive applications of the "
                        "optimal input for an n-Step Data-Driven MPC Scheme. "
                        "If not defined, defaults to the estimated system "
                        "order.")
    parser.add_argument("--controller_type", type=str,
                        default="Robust",
                        choices=["Nominal", "Robust"],
                        help="The Data-Driven MPC Controller type. Defaults "
                        "to `Robust`.")
    parser.add_argument("--eps_max", type=float, default=default_eps_max,
                        help="The estimated upper bound of the system "
                        "measurement noise. If set to zero, disables system "
                        "noise and sets a high value for the ridge "
                        "regularization base weight for alpha (`lamb_alpha`) "
                        "parameter to prevent division by zero for a Robust "
                        "Data-Driven MPC controller. Defaults to 0.002.")
    parser.add_argument("--t_sim", type=int, default=default_t_sim,
                        help="The simulation length in time steps.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for Random Number Generator "
                        "initialization to ensure reproducible results. "
                        "Defaults to `None`.")
    # Animation video output arguments
    parser.add_argument("--save_video", action='store_true', default=False,
                        help="Save generated animation as a video in an .mp4 "
                        "file using ffmpeg.")
    parser.add_argument("--video_path", type=str,
                        default=default_video_path,
                        help="Saving path for the generated animation file. "
                        "Includes file name with .mp4 extension.")
    parser.add_argument("--video_fps", type=int,
                        default=default_video_fps,
                        help="Frames per second value for the saved video.")
    parser.add_argument("--video_bitrate", type=int,
                        default=default_video_bitrate,
                        help="Bitrate value for the saved video.")
    # Verbose argument
    parser.add_argument("--verbose", type=int, default=1,
                        choices=[0, 1],
                        help="Verbose level")
    
    # TODO: Add arguments
    
    return parser.parse_args()

def main() -> None:
    # --- Parse arguments ---
    args = parse_args()
    
    # Data-Driven MPC controller arguments
    n_mpc_step = args.n_mpc_step
    controller_type_arg = controller_type_mapping[args.controller_type]
    eps_max_arg = args.eps_max

    # Simulation parameters
    t_sim = args.t_sim

    seed = args.seed

    # Animation video output arguments
    save_video = args.save_video
    video_path = args.video_path
    video_fps = args.video_fps
    video_bitrate = args.video_bitrate

    # Verbose argument
    verbose = args.verbose

    # ==============================================
    # 1. Define Simulation and Controller Parameters
    # ==============================================
    # --- Define system model (simulation) ---
    A = FourTankSysParams.A
    B = FourTankSysParams.B
    C = FourTankSysParams.C
    D = FourTankSysParams.D

    ns = A.shape[0] # System order (considered for simulation)
    m = B.shape[1] # Number of inputs
    p = C.shape[0] # Number of outputs
    
    eps_max = eps_max_arg # Estimated upper bound of the system
    # measurement noise. If set to zero, the system is considered
    # ideal with noise-free conditions, which enables testing of
    # the Nominal Data-Driven MPC controller

    # --- Define initial Input-Output data generation parameters ---
    u_range = (-1, 1) # Persistently exciting input range
    N = 400 # Initial input-output trajectory length

    # --- Define Data-Driven MPC parameters ---
    n = 4 # Estimated system order (Data-Driven MPC formulation)
    L = 30 # Prediction horizon
    Q = 3 * np.eye(p * L) # Output weighting matrix
    R = 1e-4 * np.eye(m * L) # Input weighting matrix

    # Define ridge regularization base weight for alpha, preventing
    # division by zero in noise-free conditions
    if eps_max != 0:
        lamb_alpha = 0.1 / eps_max
    else:
        lamb_alpha = 1000 # Set a high value if eps_max is zero.
    lamb_sigma = 1000 # Ridge regularization weight for sigma
    c = 100 # Convex slack variable constraint: ||sigma||_inf <= c * eps_max
    slack_var_constraint_type = SlackVarConstraintTypes.NONE # Slack
    # variable constraint type
    
    # Number of consecutive applications of the optimal input
    # for an n-Step Data-Driven MPC Scheme (multi-step)
    if n_mpc_step is None:
        n_mpc_step = n # if not defined, defaults to the estimated system
        # order, as defined in Algorithm 2 from [1].
    
    controller_type = controller_type_arg # Controller type

    # Define Input-Output equilibrium setpoint pair
    u_s = FourTankSysParams.us.reshape(-1, 1) # Control input setpoint
    # y_s = FourTankSysParams.ys.reshape(-1, 1) # System output setpoint
    y_s = calculate_output_equilibrium_setpoint(A=A, B=B, C=C, D=D, u_s=u_s)
    # Calculate the system output equilibrium setpoint from `u_s` to avoid
    # unfeasible solutions in the Nominal Data-Driven MPC Controller (see
    # module docstring for details).

    # --- Define Control Simulation parameters ---
    T = t_sim # "Closed-loop horizon" (simulation length)
    T += 1 # Add a step to calculate the last control input

    # Create a Random Number Generator for reproducibility
    np_random = np.random.default_rng(seed=seed)

    # ==============================================
    # 2. Randomize Initial System State (Simulation)
    # ==============================================
    # Randomize initial system state to demonstrate the functionality
    # of the Direct-Data Driven MPC controller
    x_i0 = np_random.random(ns) # Random system state before excitement
    x_i = np.zeros((ns, ns))
    x_i[0, :] = x_i0

    # Initialize a random input array and an output array
    u_i = np_random.uniform(*u_range, (ns, m))
    y_i = np.zeros((ns, p))

    # Generate bounded uniformly distributed additive measurement noise
    w_i = eps_max * np_random.uniform(-1, 1, (ns, p))

    # Simulate system with a random input sequence to
    # randomize its initial state
    for t in range(ns - 1):
        x_i[t + 1, :] = A @ x_i[t, :] + B @ u_i[t, :]
        y_i[t, :] = C @ x_i[t, :] + D @ u_i[t, :] + w_i[t, :]

    y_i[ns - 1, :] = C @ x_i[ns - 1, :] + D @ u_i[ns - 1, :] + w_i[ns - 1, :]

    # Calculate the initial system state
    Ot = observability_matrix(A, C)
    Tt = toeplitz_input_output_matrix(A, B, C, D, ns)
    x_0 = estimate_initial_state(Ot=Ot,
                                 Tt=Tt,
                                 U=u_i.flatten(),
                                 Y=y_i.flatten())
    
    # ====================================================
    # 3. Initial Input-Output Data Generation (Simulation)
    # ====================================================
    # Initialize system state array with random initial state
    x_d = np.zeros((N, ns))
    x_d[0, :] = x_0

    # Generate a persistently exciting input `u_d`
    # from 0 to (N - 1) in the [-1, 1] range
    u_d = np_random.uniform(*u_range, (N, m))

    # Initialize output data matrix `y_d`
    y_d = np.zeros((N, p))

    # Generate bounded uniformly distributed additive measurement noise
    w_d = eps_max * np_random.uniform(-1, 1, (N, p))

    # Simulation system with persistently exciting input `u_d`
    for k in range(N - 1):
        x_d[k + 1, :] = A @ x_d[k, :] + B @ u_d[k, :]
        y_d[k, :] = C @ x_d[k, :] + D @ u_d[k, :] + w_d[k, :]

    y_d[N - 1, :] = C @ x_d[N - 1, :] + D @ u_d[N - 1, :] + w_d[N - 1, :]
    
    # ===============================================
    # 4. Data-Driven MPC Controller Instance Creation
    # ===============================================
    # Create a Direct Data-Driven MPC controller
    if verbose:
        print(f"Initializing {controller_type.name.capitalize()} "
              "Data-Driven MPC controller")
    
    direct_data_driven_mpc_controller = DirectDataDrivenMPCController(
        n=n,
        m=m,
        p=p,
        u_d=u_d,
        y_d=y_d,
        L=L,
        Q=Q,
        R=R,
        u_s=u_s,
        y_s=y_s,
        eps_max=eps_max,
        lamb_alpha=lamb_alpha,
        lamb_sigma=lamb_sigma,
        c=c,
        slack_var_constraint_type=slack_var_constraint_type,
        controller_type=controller_type)

    # ===============================
    # 5. Data-Driven MPC Control Loop
    # ===============================
    # Initialize control loop state array
    x_sys = np.zeros((T, ns))
    x_sys[0, :] = x_d[-1, :] # Set initial state to the last state of x_d
    # to ensure continuity with the previous input-output sequence

    # Initialize control loop input-output arrays
    u_sys = np.zeros((T, m))
    y_sys = np.zeros((T, p))

    # Generate bounded uniformly distributed additive measurement noise
    w_sys = eps_max * np_random.uniform(-1, 1, (T, p))
    

    # --- Simulate Data-Driven MPC control system ---
    # Simulate the Data-Driven MPC control system following Algorithm 1 for a
    # Data-Driven MPC Scheme, and Algorithm 2 for an n-Step Data-Driven MPC
    # Scheme, as described in [1].
    if verbose:
        print("Starting Data-Driven MPC control system simulation")
    
    for t in range(0, T, n_mpc_step):
        # --- Algorithm 1 and Algorithm 2 (n-step): ---
        # 1) Solve Data-Driven MPC after taking past `n` input-output
        #    measurements u[t-n, t-1], y[t-n, t-1].

        # Update and solve the Data-Driven MPC problem
        direct_data_driven_mpc_controller.update_and_solve_data_driven_mpc()

        # Simulate closed loop
        for k in range(t, min(t + n_mpc_step, T - 1)):
            # --- Algorithm 1: ---
            # 2) Apply the input ut = ubar*[0](t).
            # --- Algorithm 2 (n-step): ---
            # 2) Apply the input sequence u[t, t+n-1] = ubar*[0, n-1](t)
            #    over the next `n` time steps. 

            # Update control input
            n_step = k - t # Time step `n`. Results 0 for n_mpc_step = 1
            optimal_u_step_n = (
                direct_data_driven_mpc_controller.get_optimal_control_input_at_step(n_step=n_step))
            u_sys[k, :] = optimal_u_step_n
            
            # --- Simulate system with optimal control input ---
            x_sys[k + 1, :] = A @ x_sys[k, :] + B @ u_sys[k, :]
            y_sys[k, :] = C @ x_sys[k, :] + D @ u_sys[k, :] + w_sys[k, :]
            
            # --- Algorithm 1 and Algorithm 2 (n-step): ---
            # 1) At time `t`, take the past `n` measurements u[t-n, t-1],
            #    y[t-n, t-1] and solve Data-Driven MPC.
            # * Data-Driven MPC is solved at the start of the next iteration.

            # Update past input-output measurements
            direct_data_driven_mpc_controller.store_input_output_measurement(
                u_current=u_sys[k, :].reshape(-1, 1),
                y_current=y_sys[k, :].reshape(-1, 1)
            )

        # --- Algorithm 1: ---
        # 3) Set t = t + 1 and go back to 1).
        # --- Algorithm 2 (n-step): ---
        # 3) Set t = t + n and go back to 1).

        if verbose:
            # Get current step MPC cost value
            mpc_cost_val = (
                direct_data_driven_mpc_controller.get_optimal_cost_value())
            # Calculate output error
            y_error = y_s.flatten() - y_sys[k, :].flatten()
            # Format output error
            formatted_y_error = ', '.join(
                [f'e_{i} = {error:.3f}' for i, error in enumerate(y_error)])
            # Print time step, MPC cost value, and formatted error
            print(f"Time step: {t:>4} - MPC cost value: {mpc_cost_val:>8.4f}"
                  f" - Error: {formatted_y_error}")
        
    # Remove last step from simulation data (added
    # previously to calculate the last control input)
    u_sys = u_sys[0:-1]
    y_sys = y_sys[0:-1]
    T -= 1

    # =====================================================
    # 6. Plot and Animate Control System Inputs and Outputs
    # =====================================================
    # --- Plot control system inputs and outputs ---
    if verbose:
        print("Displaying control system inputs and outputs plot")
    
    plot_input_output(u_k=u_sys,
                      y_k=y_sys,
                      u_s=u_s,
                      y_s=y_s,
                      figsize=(14, 8),
                      dpi=100)
    
    # --- Plot data including initial input-output sequences ---
    # Create data arrays including initial input-output data used for
    # the data-driven characterization of the system
    U = np.vstack([u_d, u_sys])
    Y = np.vstack([y_d, y_sys])

    # Plot extended input-output data
    if verbose:
        print("Displaying control system inputs and outputs including "
              "initial input-output measured data")
    
    plot_input_output(u_k=U,
                      y_k=Y,
                      u_s=u_s,
                      y_s=y_s,
                      initial_steps=N,
                      figsize=(14, 8),
                      dpi=100)

    # --- Animate extended input-output data ---
    if verbose:
        print("Displaying animation from extended input-output data")
    
    ani = plot_input_output_animation(u_k=U,
                                      y_k=Y,
                                      u_s=u_s,
                                      y_s=y_s,
                                      initial_steps=N,
                                      figsize=(14, 8),
                                      dpi=100,
                                      interval=1)
    plt.show() # Show animation
    
    if save_video:
        if verbose:
            print('Saving input-output animation as an MP4 video')
            print(f'Animation video path: {video_path}')
        
        # Save input-output animation as an MP4 video
        save_animation(animation=ani,
                       total_frames=N + T,
                       fps=video_fps,
                       bitrate=video_bitrate,
                       file_path=video_path)
        
        if verbose:
            print('Animation MP4 video saved.')

    plt.close() # Close figures

if __name__ == "__main__":
    main()
    