"""
Robust Data-Driven Model Predictive Control (MPC) Reproduction

This script implements a reproduction of the example presented by J. Berberich
et al. in Section V of [1], which illustrates various Robust Data-Driven MPC
controller schemes applied to a four-tank system model. The implemented
controller schemes are:
    - 1-step Robust Data-Driven MPC scheme with terminal equality constraints
        (TEC)
    - n-step Robust Data-Driven MPC scheme  with terminal equality constraints
        (TEC, n-step)
    - 1-step Robust Data-Driven MPC scheme without terminal equality
        constraints (UCON)

The implementation uses the parameters defined in the paper example, including
those for the system model, the initial input-output data generation, and the
Data-Driven MPC controller setup. Additionally, the initial output of the
system is set to `y_0 = [0.4, 0.4]` to reproduce the closed-loop output graphs
shown in Fig. 2 of [1].

Notes:
    A default seed is provided for the random number generator to closely
    match the results presented in the example. This seed can be modified
    through argument parsing; however, since the closed loop of the Robust
    Data-Driven MPC scheme without terminal equality constraints is unstable
    and diverges, using different seeds might result in infeasible solutions.
    In contrast, the schemes with terminal equality constraints do not exhibit
    this issue.

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

from utilities.controller.controller_creation import (
    get_data_driven_mpc_controller_params)
from utilities.controller.controller_operation import (
    randomize_initial_system_state, simulate_n_input_output_measurements,
    generate_initial_input_output_data)
from utilities.reproduction.paper_reproduction import (
    DataDrivenMPCScheme, get_equilibrium_state_from_output,
    create_data_driven_mpc_controllers_reproduction,
    simulate_data_driven_mpc_control_loops_reproduction,
    plot_input_output_reproduction)

from utilities.model_simulation import LTISystemModel

# Directory paths
dirname = os.path.dirname
project_dir = dirname(dirname(__file__))
examples_dir = os.path.join(project_dir, 'examples')
models_config_dir = os.path.join(examples_dir, 'config', 'models')
controller_config_dir = os.path.join(examples_dir, 'config', 'controllers')

# Model configuration file
model_config_file = 'four_tank_system_params.yaml'
model_config_path = os.path.join(models_config_dir,
                                 model_config_file)
model_key_value = 'FourTankSystem'

# Data-Driven MPC controller configuration file
controller_config_file = 'data_driven_mpc_example_params.yaml'
controller_config_path = os.path.join(controller_config_dir,
                                      controller_config_file)
controller_key_value = 'data_driven_mpc_params'

# Simulation parameters
default_t_sim = 600 # Default simulation length in time steps
default_seed = 4 # Default seed for the RNG

# Paper reproduction parameters
y_0 = [0.4, 0.4] # Initial system output for reproduction
u_ylimits = [[-15.0, 15.0], [-15.0, 15.0]] # Control input plot Y-axis limits
y_ylimits = [[0.4, 1.0], [0.4, 1.0]] # System output plot Y-axis limits

# Robust Data-Driven MPC controllers showcased in paper example
dd_mpc_controller_schemes = [DataDrivenMPCScheme.TEC,
                             DataDrivenMPCScheme.TEC_N_STEP,
                             DataDrivenMPCScheme.UCON]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data-Driven MPC "
                                     "Controller Reproduction")
    parser.add_argument("--t_sim", type=int, default=default_t_sim,
                        help="The simulation length in time steps.")
    parser.add_argument("--seed", type=int, default=default_seed,
                        help="Seed for Random Number Generator "
                        "initialization to ensure reproducible results. "
                        "Defaults to 0.")
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

    # Simulation parameters
    t_sim = args.t_sim
    seed = args.seed

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
        verbose=verbose)

    # --- Define Control Simulation parameters ---
    n_steps = t_sim + 1 # Number of simulation steps

    # Create a Random Number Generator for reproducibility
    np_random = np.random.default_rng(seed=seed)

    if verbose:
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
        formatted_schemes = ', '.join([scheme.name for scheme
                                       in dd_mpc_controller_schemes])
        print("Initializing Robust Data-Driven MPC controllers following "
              f"schemes: {formatted_schemes}")

    # Create Direct Data-Driven MPC controllers for each scheme
    dd_mpc_controllers = create_data_driven_mpc_controllers_reproduction(
        controller_config=dd_mpc_config,
        u_d=u_d,
        y_d=y_d,
        data_driven_mpc_controller_schemes=dd_mpc_controller_schemes)
    
    # ========================================================
    # 5. Set Initial System State from Output for Reproduction
    # ========================================================
    if verbose:
        print("Setting initial system output to y_0 = [0.4, 0.4] for "
              "reproduction")

    # To set the initial system output (y_0 = [0.4, 0.4]) to reproduce the
    # results from the paper example, we estimate the initial system state
    # corresponding to `y_0` using the system equations and update the model
    # to this state.
    # 
    # This is only used for reproduction and does not represent the typical
    # operation of a Direct Data-Driven MPC controller, as this type of
    # controllers are designed to control unknown systems without prior system
    # identification.
    
    # Estimate the initial state from equilibrium input-output trajectory
    xrep_0 = get_equilibrium_state_from_output(system_model=system_model,
                                               y_eq=y_0)
    
    # Set the system state corresponding to the initial
    # input-output pair for reproduction
    system_model.set_state(xrep_0)

    if verbose > 1:
        print(f"    Initial system state set to: {xrep_0} based on "
              "equilibrium output")

    # =============================================================
    # 6. Initial Simulation to Store Past Input-Output Measurements
    # =============================================================
    if verbose:
        print("Simulating `n` steps using control input setpoint to "
              "update controllers' past input-output measurements")

    # Simulate `n` (the estimated system order) steps of the system using a
    # constant input (the controller's input setpoint). The resulting
    # input-output trajectory is then used to update the past `n` input-output
    # measurements in the Data-Driven MPC controllers. This is important to
    # account for the change in the system state considered for reproduction.
    #
    # Typically, a Data-Driven MPC controller would use the last `n`
    # measurements from the initial input-output data used to characterize the
    # unknown system as the past `n` input-output measurements.
    U_n, Y_n = simulate_n_input_output_measurements(
        system_model=system_model,
        controller_config=dd_mpc_config,
        np_random=np_random)

    # Update the past `n` input-output measurements of each
    # Data-Driven MPC controller (for reproduction)
    for controller in dd_mpc_controllers:
        controller.set_past_input_output_data(
            u_past=U_n.reshape(-1, 1),
            y_past=Y_n.reshape(-1, 1))
        
    if verbose:
        print("Controllers' past `n` input-output measurements updated")
        if verbose > 1:
            print(f"    Past input data = {U_n.shape}, Past output data = "
                  f"{Y_n.shape}")

    # ===============================
    # 7. Data-Driven MPC Control Loop
    # ===============================
    if verbose:
        print("Starting control system simulation for each Robust "
              "Data-Driven MPC scheme")

    # Simulate the Data-Driven MPC control systems following Algorithm 1 for a
    # Data-Driven MPC Scheme, and Algorithm 2 for an n-Step Data-Driven MPC
    # Scheme, as described in [1].
    #
    # Simulate from `n` to `T - n`, considering that the first `n` steps are
    # used to store the past `n` input-output measurements for each
    # controller. Here, `n` is the estimated system order from the Data-Driven
    # MPC controller configuration.
    n = dd_mpc_config['n'] # Estimated system order (Data-Driven MPC)
    u_sys_data, y_sys_data = (
        simulate_data_driven_mpc_control_loops_reproduction(
            system_model=system_model,
            data_driven_mpc_controllers=dd_mpc_controllers,
            n_steps=n_steps - n,
            np_random=np_random,
            verbose=verbose))
    
    # Construct input-output data from 0 to `T - 1`
    for i in range(len(dd_mpc_controllers)):
        u_sys_data[i] = np.vstack([U_n, u_sys_data[i]])
        y_sys_data[i] = np.vstack([Y_n, y_sys_data[i]])

    # =========================================
    # 8. Plot Control System Inputs and Outputs
    # =========================================
    u_s = dd_mpc_config['u_s'] # Control input setpoint
    y_s = dd_mpc_config['y_s'] # System output setpoint

    # --- Plot control system inputs and outputs ---
    if verbose:
        print("Displaying control system inputs and outputs plot")

    plot_input_output_reproduction(
        data_driven_mpc_controller_schemes=dd_mpc_controller_schemes,
        u_data=u_sys_data,
        y_data=y_sys_data,
        u_s=u_s,
        y_s=y_s,
        u_ylimits=u_ylimits,
        y_ylimits=y_ylimits,
        figsize=(12, 8),
        dpi=100,
        title="Robust Data-Driven MPC Reproduction")
    
    plt.close() # Close figures

if __name__ == "__main__":
    main()
    