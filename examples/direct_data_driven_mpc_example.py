"""
Direct Data-Driven Model Predictive Control (MPC) Example Script

This script demonstrates the setup, simulation, and visualization of a Direct
Data-Driven MPC controller applied to a four-tank system model based on
research by J. Berberich et al. [1]. The implementation follows the parameters
defined in the example presented in Section V of [1], including those for the
system model, the initial input-output data generation, and the Data-Driven
MPC controller setup.

To illustrate a typical controller operation, this script does not set the
initial system output to `y_0 = [0.4, 0.4]`, as shown in the closed-loop
output graphs from Fig. 2 in [1]. Instead, the initial system state is
estimated using a randomized input sequence.

For a closer approximation of the results presented in the paper's example,
which assumes the initial system output `y_0 = [0.4, 0.4]`, please refer to
'robust_data_driven_mpc_reproduction.py'.

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

from examples.utilities.controller_creation import (
    get_data_driven_mpc_controller_params, create_data_driven_mpc_controller)
from examples.utilities.controller_operation import (
    randomize_initial_system_state, generate_initial_input_output_data,
    simulate_data_driven_mpc_control_loop)

from utilities.data_visualization import (
    plot_input_output, plot_input_output_animation, save_animation)

from direct_data_driven_mpc.direct_data_driven_mpc_controller import (
    DataDrivenMPCType, SlackVarConstraintTypes)

from utilities.model_simulation import LTISystemModel

from examples.utilities.plot_styles import (
    INPUT_OUTPUT_PLOT_PARAMS, INPUT_OUTPUT_PLOT_PARAMS_SMALL)

# Directory paths
dirname = os.path.dirname
project_dir = dirname(dirname(__file__))
examples_dir = os.path.join(project_dir, 'examples')
models_config_dir = os.path.join(examples_dir, 'config', 'models')
controller_config_dir = os.path.join(examples_dir, 'config', 'controllers')
default_animation_dir = os.path.join(project_dir, 'animation_outputs')

# Model configuration file
default_model_config_file = 'four_tank_system_params.yaml'
default_model_config_path = os.path.join(models_config_dir,
                                         default_model_config_file)
default_model_key_value = 'FourTankSystem'

# Data-Driven MPC controller configuration file
default_controller_config_file = 'data_driven_mpc_example_params.yaml'
default_controller_config_path = os.path.join(controller_config_dir,
                                              default_controller_config_file)
default_controller_key_value = 'data_driven_mpc_params'

# Animation video default parameters
default_video_name = "data-driven_mpc_sim.mp4"
default_video_path = os.path.join(default_animation_dir, default_video_name)
default_video_fps = 100
default_video_bitrate = 1800

# Data-Driven MPC controller parameters
controller_type_mapping = {
    "Nominal": DataDrivenMPCType.NOMINAL,
    "Robust": DataDrivenMPCType.ROBUST,
}
slack_var_constraint_type_mapping = {
    "NonConvex": SlackVarConstraintTypes.NON_CONVEX,
    "Convex": SlackVarConstraintTypes.CONVEX,
    "None": SlackVarConstraintTypes.NONE
}
default_t_sim = 600 # Default simulation length in time steps

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Data-Driven MPC "
                                     "Controller Example")
    # Model configuration file arguments
    parser.add_argument("--model_config_path", type=str,
                        default=default_model_config_path,
                        help="The path to the YAML configuration file "
                        "containing the model parameters.")
    parser.add_argument("--model_key_value", type=str,
                        default=default_model_key_value,
                        help="The key to access the model parameters in the "
                        "configuration file.")
    # Data-Driven MPC controller configuration file arguments
    parser.add_argument("--controller_config_path", type=str,
                        default=default_controller_config_path,
                        help="The path to the YAML configuration file "
                        "containing the Data-Driven MPC controller "
                        "parameters.")
    parser.add_argument("--controller_key_value", type=str,
                        default=default_controller_key_value,
                        help="The key to access the Data-Driven MPC "
                        "controller parameters in the configuration file.")
    # Data-Driven MPC controller arguments
    parser.add_argument("--n_mpc_step", type=int,
                        default=None,
                        help="The number of consecutive applications of the "
                        "optimal input for an n-Step Data-Driven MPC Scheme.")
    parser.add_argument("--controller_type", type=str,
                        default=None,
                        choices=["Nominal", "Robust"],
                        help="The Data-Driven MPC Controller type.")
    parser.add_argument("--slack_var_const_type", type=str,
                        default=None,
                        choices=["None", "Convex", "NonConvex"],
                        help="The constraint type for the slack variable "
                        "`sigma` in a Robust Data-Driven MPC formulation.")
    parser.add_argument("--eps_max", type=float, default=None,
                        help="The estimated upper bound of the system "
                        "measurement noise. If set to zero, disables system "
                        "noise and sets a high value for the ridge "
                        "regularization base weight for alpha (`lamb_alpha`) "
                        "parameter to prevent division by zero for a Robust "
                        "Data-Driven MPC controller.")
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
    parser.add_argument("--verbose", type=int, default=2,
                        choices=[0, 1, 2],
                        help="The verbosity level: 0 = no output, 1 = "
                        "minimal output, 2 = detailed output.")
    
    # TODO: Add arguments
    
    return parser.parse_args()

def main() -> None:
    # --- Parse arguments ---
    args = parse_args()

    # Model parameters
    model_config_path = args.model_config_path
    model_key_value = args.model_key_value

    # Data-Driven MPC controller parameters
    controller_config_path = args.controller_config_path
    controller_key_value = args.controller_key_value
    
    # Data-Driven MPC controller arguments
    n_mpc_step = args.n_mpc_step
    controller_type_arg = args.controller_type
    slack_var_const_type_arg = args.slack_var_const_type
    eps_max = args.eps_max

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
    if verbose:
        print("Loading system parameters from configuration file")

    system_model = LTISystemModel(config_file=model_config_path,
                                  model_key_value=model_key_value,
                                  verbose=verbose)

    # Override model parameters with parsed arguments
    if (eps_max is not None):
        if verbose:
            print("Overriding model parameters")

    # Override the upper bound of the system measurement noise
    # with parsed argument if passed
    if eps_max is not None:
        system_model.set_eps_max(eps_max=eps_max)

        if verbose > 1:
            if eps_max == 0:
                print("    System set to ideal, noise-free conditions")
            else:
                print("    Upper bound of system measurement noise set to: "
                      f"{eps_max}")
    
    # If set to zero, the system is considered ideal with noise-free
    # conditions, enabling testing of the Nominal Data-Driven MPC controller

    # --- Define Data-Driven MPC Controller Parameters ---
    if verbose:
        print("Loading Data-Driven MPC controller parameters from "
              "configuration file")

    # Load Data-Driven MPC controller parameters from configuration file
    m = system_model.get_number_inputs() # Number of inputs
    p = system_model.get_number_outputs() # Number of outputs
    dd_mpc_config = get_data_driven_mpc_controller_params(
        config_file=controller_config_path,
        controller_key_value=controller_key_value,
        m=m,
        p=p,
        eps_bar=eps_max,
        verbose=verbose)
    
    # Override controller parameters with parsed arguments
    if (n_mpc_step is not None or controller_type_arg is not None
        or slack_var_const_type_arg is not None):
        if verbose:
            print("Overriding Data-Driven MPC controller parameters")
    
    # Override the number of consecutive applications of the
    # optimal input (n-Step Data-Driven MPC Scheme (multi-step))
    # with parsed argument if passed
    if n_mpc_step is not None:
        dd_mpc_config['n_mpc_step'] = n_mpc_step

        if verbose > 1:
            print("    n-Step Data-Driven MPC scheme parameter "
                  f"(`n_mpc_step`) set to: {n_mpc_step}")
    
    # Override the Controller type with parsed argument if passed
    if controller_type_arg is not None:
        dd_mpc_config['controller_type'] = controller_type_mapping[
            controller_type_arg]
        
        if verbose > 1:
            print("    Data-Driven MPC controller type set to: "
                  f"{dd_mpc_config['controller_type'].name}")
    
    # Override the slack variable constraint type
    # with parsed argument if passed
    if slack_var_const_type_arg is not None:
        dd_mpc_config['slack_var_constraint_type'] = (
            slack_var_constraint_type_mapping[slack_var_const_type_arg])
        
        if verbose > 1:
            print("    Slack variable constraint type set to: "
                  f"{dd_mpc_config['slack_var_constraint_type'].name}")

    # --- Define Control Simulation parameters ---
    n_steps = t_sim + 1 # Number of simulation steps

    # Create a Random Number Generator for reproducibility
    np_random = np.random.default_rng(seed=seed)

    if verbose:
        if seed is None:
            print("Random number generator initialized with a random seed")
        else:
            print(f"Random number generator initialized with seed: {seed}")

    # ==============================================
    # 2. Randomize Initial System State (Simulation)
    # ==============================================
    if verbose:
        print(f"Randomizing initial system state")

    # Randomize the initial internal state of the system to ensure
    # the model starts in a plausible random state
    x_0 = randomize_initial_system_state(system_model=system_model,
                                         controller_config=dd_mpc_config,
                                         np_random=np_random)
    
    # Set system state to the estimated plausible random initial state
    system_model.set_state(state=x_0)

    if verbose > 1:
        print(f"    Initial system state set to: {x_0}")

    # ====================================================
    # 3. Initial Input-Output Data Generation (Simulation)
    # ====================================================
    if verbose:
        print("Generating initial input-output data")

    # Generate initial input-output data using a
    # generated persistently exciting input
    u_d, y_d = generate_initial_input_output_data(
        system_model=system_model,
        controller_config=dd_mpc_config,
        np_random=np_random)
    
    if verbose > 1:
        print(f"    Input data shape: {u_d.shape}, Output data shape: "
              f"{y_d.shape}")

    # ===============================================
    # 4. Data-Driven MPC Controller Instance Creation
    # ===============================================
    if verbose:
        controller_type = dd_mpc_config['controller_type']
        print(f"Initializing {controller_type.name.capitalize()} "
              "Data-Driven MPC controller")

    # Create a Direct Data-Driven MPC controller
    dd_mpc_controller = create_data_driven_mpc_controller(
        controller_config=dd_mpc_config, u_d=u_d, y_d=y_d)

    # ===============================
    # 5. Data-Driven MPC Control Loop
    # ===============================
    if verbose:
        print(f"Starting {controller_type.name.capitalize()} Data-Driven "
              "MPC control system simulation")

    # Simulate the Data-Driven MPC control system following Algorithm 1 for a
    # Data-Driven MPC Scheme, and Algorithm 2 for an n-Step Data-Driven MPC
    # Scheme, as described in [1].
    u_sys, y_sys = simulate_data_driven_mpc_control_loop(
        system_model=system_model,
        data_driven_mpc_controller=dd_mpc_controller,
        n_steps=n_steps,
        np_random=np_random,
        verbose=verbose)

    # =====================================================
    # 6. Plot and Animate Control System Inputs and Outputs
    # =====================================================
    N = dd_mpc_config['N'] # Initial input-output trajectory length
    u_s = dd_mpc_config['u_s'] # Control input setpoint
    y_s = dd_mpc_config['y_s'] # System output setpoint

    # --- Plot control system inputs and outputs ---
    if verbose:
        print("Displaying control system inputs and outputs plot")
    
    plot_input_output(u_k=u_sys,
                      y_k=y_sys,
                      u_s=u_s,
                      y_s=y_s,
                      figsize=(14, 8),
                      dpi=100,
                      **INPUT_OUTPUT_PLOT_PARAMS)
    
    # --- Plot data including initial input-output sequences ---
    # Create data arrays including initial input-output data used for
    # the data-driven characterization of the system
    U = np.vstack([u_d, u_sys])
    Y = np.vstack([y_d, y_sys])

    # Plot extended input-output data
    if verbose:
        print("Displaying control system inputs and outputs including "
              "initial input-output measurements")
    
    plot_input_output(u_k=U,
                      y_k=Y,
                      u_s=u_s,
                      y_s=y_s,
                      initial_steps=N,
                      figsize=(14, 8),
                      dpi=100,
                      **INPUT_OUTPUT_PLOT_PARAMS_SMALL)

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
                                      interval=1,
                                      **INPUT_OUTPUT_PLOT_PARAMS_SMALL)
    plt.show() # Show animation
    
    if save_video:
        if verbose:
            print("Saving extended input-output animation as an MP4 video")
            if verbose > 1:
                print(f"    Saving video to: {video_path}")
                print(f"    Video FPS: {video_fps}, Bitrate: "
                      f"{video_bitrate}, Total Frames: {N + n_steps}")
        
        # Save input-output animation as an MP4 video
        save_animation(animation=ani,
                       total_frames=N + n_steps,
                       fps=video_fps,
                       bitrate=video_bitrate,
                       file_path=video_path)
        
        if verbose:
            print("Animation MP4 video saved successfully")

    plt.close() # Close figures

if __name__ == "__main__":
    main()
    