from typing import Tuple

import numpy as np
from numpy.random import Generator

from utilities.model_simulation import LTIModel

from utilities.controller.controller_creation import (
    DataDrivenMPCParamsDictType)
from direct_data_driven_mpc.direct_data_driven_mpc_controller import (
    DirectDataDrivenMPCController)

def randomize_initial_system_state(
    system_model: LTIModel,
    controller_config: DataDrivenMPCParamsDictType,
    np_random: Generator
) -> np.ndarray:
    """
    Randomly generate a plausible initial state for a Linear Time-Invariant
    (LTI) system model.

    This function initializes the system state with random values within the
    [-1, 1] range. Afterward, it simulates the system using random input and
    noise sequences to generate an input-output trajectory, which is then used
    to estimate the initial system state.
    
    Note:
        The random input sequence is generated based on the `u_range`
        parameter from the controller configuration (`controller_config`). The
        noise sequence is generated considering the defined noise bounds from
        the system.
    
    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        controller_config (DataDrivenMPCParamsDictType): A dictionary
            containing Data-Driven MPC controller parameters, including the
            range of the persistently exciting input (`u_range`).
        np_random (Generator): A Numpy random number generator for generating 
            the random initial system state, persistently exciting input, and
            system output noise.
    
    Returns:
        np.ndarray: A vector of shape `(n, )` representing the estimated initial
            state of the system, where `n` is the system's order.
    """
    # Retrieve model parameters
    ns = system_model.get_system_order() # System order (simulation)
    m = system_model.get_number_inputs() # Number of inputs
    p = system_model.get_number_outputs() # Number of outputs
    eps_max_sim = system_model.get_eps_max() # Upper bound of the system
    # measurement noise (simulation)

    # Retrieve Data-Driven MPC controller parameters
    u_range = controller_config['u_range'] # Range of the persistently
    # exciting input u_d
    
    # Randomize initial system state before excitation
    x_i0 = np_random.uniform(-1.0, 1.0, size=ns)
    system_model.set_state(state=x_i0)

    # Generate a random input array
    u_i = np_random.uniform(*u_range, (ns, m))

    # Generate bounded uniformly distributed additive measurement noise
    w_i = eps_max_sim * np_random.uniform(-1.0, 1.0, (ns, p))

    # Simulate the system with the generated random input and noise
    # sequences to obtain output data
    y_i = system_model.simulate(U=u_i, W=w_i, steps=ns)

    # Calculate the initial state of the system
    # from the input-output trajectory
    x_0 = system_model.get_initial_state_from_trajectory(
        U=u_i.flatten(), Y=y_i.flatten())
    
    return x_0

def generate_initial_input_output_data(
    system_model: LTIModel,
    controller_config: DataDrivenMPCParamsDictType,
    np_random: Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate input-output trajectory data from a system using Data-Driven MPC
    controller parameters.

    This function generates a persistently exciting input `u_d` and random
    noise based on the specified controller and system parameters. Then, it
    simulates the system using these input and noise sequences to generate the
    output reponse `y_d`. The resulting `u_d` and `y_d` arrays represent the
    input-output trajectory data measured from the system, which is necessary
    for system characterization in a Data-Driven MPC formulation.

    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        controller_config (DataDrivenMPCParamsDictType): A dictionary
            containing Data-Driven MPC controller parameters, including the
            initial input-output trajectory length (`N`) and the range of the
            persistently exciting input (`u_range`).
        np_random (Generator): A Numpy random number generator for generating
            the persistently exciting input and random noise for the system's
            output.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays: a
            persistently exciting input (`u_d`) and the system's output
            response (`y_d`). The input array has shape `(N, m)` and the
            output array has shape `(N, p)`, where `N` is the trajectory
            length, `m` is the number of control inputs, and `p` is the number
            of system outputs.
    """
    # Retrieve model parameters
    m = system_model.get_number_inputs() # Number of inputs
    p = system_model.get_number_outputs() # Number of outputs
    eps_max_sim = system_model.get_eps_max() # Upper bound of the system
    # measurement noise (simulation)

    # Retrieve Data-Driven MPC controller parameters
    N = controller_config['N'] # Initial input-output trajectory length
    u_range = controller_config['u_range'] # Range of the persistently
    # exciting input u_d

    # Generate a persistently exciting input `u_d` from 0 to (N - 1)
    u_d = np_random.uniform(*u_range, (N, m))

    # Generate bounded uniformly distributed additive measurement noise
    w_d = eps_max_sim * np_random.uniform(-1.0, 1.0, (N, p))

    # Simulate the system with the persistently exciting input `u_d` and
    # the generated noise sequence to obtain output data
    y_d = system_model.simulate(U=u_d, W=w_d, steps=N)

    return u_d, y_d

def simulate_n_input_output_measurements(
    system_model: LTIModel,
    controller_config: DataDrivenMPCParamsDictType,
    np_random: Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a control input setpoint applied to a system over `n` (the
    estimated system order) time steps and return the resulting input-output
    data sequences.

    This function retrieves the control input setpoint (`u_s`) and the
    estimated system order (`n`) from a Data-Driven MPC controller
    configuration. Then, it simulates the system using a constant input `u_s`
    and random output noise over `n` time steps. The resulting input-output
    trajectory can be used to update the past `n` input-output measurements
    of a previously initialized Data-Driven MPC controller, allowing it to
    operate on a system with a different state.
    
    Note:
        This function is used for scenarios where a Data-Driven MPC controller
        has been initialized but needs to be adjusted to match different
        system states.

    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        controller_config (DataDrivenMPCParamsDictType): A dictionary
            containing Data-Driven MPC controller configuration parameters,
            including the estimated system order (`n`) and the control input
            setpoint (`u_s`).
        np_random (Generator): A Numpy random number generator for generating
            random noise for the system's output.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - An array of shape `(n, m)` representing the constant input
                setpoint applied to the system over `n` time steps, where `n`
                is the system order and `m` is the number of control inputs.
            - An array of shape `(n, p)` representing the output response of
                the system, where `n` is the system order and `p` is the
                number of system outputs.
    """
    # Retrieve model parameters
    m = system_model.get_number_inputs() # Number of inputs
    p = system_model.get_number_outputs() # Number of outputs
    eps_max_sim = system_model.get_eps_max() # Upper bound of the system
    # measurement noise

    # Retrieve Data-Driven MPC controller parameters
    n = controller_config['n'] # Estimated system order
    u_s = controller_config['u_s'] # Control input setpoint

    # Construct input array from controller's input setpoint
    U_n = np.tile(u_s, (n, 1)).reshape(n, m)

    # Generate bounded uniformly distributed additive measurement noise
    W_n = eps_max_sim * np_random.uniform(-1.0, 1.0, (n, p))
    
    # Simulate the system with the constant input and generated
    # noise sequences
    Y_n = system_model.simulate(U=U_n, W=W_n, steps=n)
    
    return U_n, Y_n

def simulate_data_driven_mpc_control_loop(
    system_model: LTIModel,
    data_driven_mpc_controller: DirectDataDrivenMPCController,
    n_steps: int,
    np_random: Generator,
    verbose: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a Data-Driven MPC control loop applied to a system and return the
    resulting input-output data sequences.

    This function simulates the operation of a Data-Driven MPC controller to
    control a system in closed-loop, following the Data-Driven MPC schemes
    described in Algorithm 1 (Nominal) and Algorithm 2 (Robust) from [1].
    
    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        data_driven_mpc_controller (DirectDataDrivenMPCController): A
            `DirectDataDrivenMPCController` instance representing a
            Data-Driven MPC controller.
        n_steps (int): The number of time steps for the simulation.
        np_random (Generator): A Numpy random number generator for generating
            random noise for the system's output.
        verbose (int): The verbosity level: 0 = no output, 1 = minimal output,
            2 = detailed output.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - An array of shape `(n_steps, m)` representing the optimal control
                inputs applied to the system, where `m` is the number of
                control inputs.
            - An array of shape `(n_steps, p)` representing the output response
                of the system, where `p` is the number of system outputs.

    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """
    # Retrieve model parameters
    m = system_model.get_number_inputs() # Number of inputs
    p = system_model.get_number_outputs() # Number of outputs
    eps_max_sim = system_model.get_eps_max() # Upper bound of the system
    # measurement noise (simulation)

    # Retrieve Data-Driven MPC controller parameters
    # Control input setpoint
    u_s = data_driven_mpc_controller.u_s
    # System output setpoint
    y_s = data_driven_mpc_controller.y_s
    # Number of consecutive applications of the optimal input
    # for an n-Step Data-Driven MPC Scheme (multi-step)
    n_mpc_step = data_driven_mpc_controller.n_mpc_step

    # Initialize control loop input-output data arrays
    u_sys = np.zeros((n_steps, m))
    y_sys = np.zeros((n_steps, p))

    # Generate bounded uniformly distributed additive measurement noise
    w_sys = eps_max_sim * np_random.uniform(-1.0, 1.0, (n_steps, p))

    # --- Simulate Data-Driven MPC control system ---
    # Simulate the Data-Driven MPC control system following Algorithm 1 for a
    # Data-Driven MPC Scheme, and Algorithm 2 for an n-Step Data-Driven MPC
    # Scheme, as described in [1].
    for t in range(0, n_steps, n_mpc_step):
        # --- Algorithm 1 and Algorithm 2 (n-step): ---
        # 1) Solve Data-Driven MPC after taking past `n` input-output
        #    measurements u[t-n, t-1], y[t-n, t-1].

        # Update and solve the Data-Driven MPC problem
        data_driven_mpc_controller.update_and_solve_data_driven_mpc()

        # Simulate closed loop
        for k in range(t, min(t + n_mpc_step, n_steps)):
            # --- Algorithm 1: ---
            # 2) Apply the input ut = ubar*[0](t).
            # --- Algorithm 2 (n-step): ---
            # 2) Apply the input sequence u[t, t+n-1] = ubar*[0, n-1](t)
            #    over the next `n` time steps. 

            # Update control input
            n_step = k - t # Time step `n`. Results 0 for n_mpc_step = 1
            optimal_u_step_n = (
                data_driven_mpc_controller.get_optimal_control_input_at_step(
                    n_step=n_step))
            u_sys[k, :] = optimal_u_step_n
            
            # --- Simulate system with optimal control input ---
            y_sys[k, :] = system_model.simulate_step(u=u_sys[k, :],
                                                     w=w_sys[k, :])
            
            # --- Algorithm 1 and Algorithm 2 (n-step): ---
            # 1) At time `t`, take the past `n` measurements u[t-n, t-1],
            #    y[t-n, t-1] and solve Data-Driven MPC.
            # * Data-Driven MPC is solved at the start of the next iteration.

            # Update past input-output measurements
            data_driven_mpc_controller.store_input_output_measurement(
                u_current=u_sys[k, :].reshape(-1, 1),
                y_current=y_sys[k, :].reshape(-1, 1)
            )

        # --- Algorithm 1: ---
        # 3) Set t = t + 1 and go back to 1).
        # --- Algorithm 2 (n-step): ---
        # 3) Set t = t + n and go back to 1).

        if verbose > 1:
            # Get current step MPC cost value
            mpc_cost_val = (
                data_driven_mpc_controller.get_optimal_cost_value())
            # Calculate input and output errors
            u_error = u_s.flatten() - u_sys[k, :].flatten()
            y_error = y_s.flatten() - y_sys[k, :].flatten()
            # Format error arrays for printing
            formatted_u_error = ', '.join([f'u_{i + 1}e = {error:>6.3f}'
                                           for i, error
                                           in enumerate(u_error)])
            formatted_y_error = ', '.join([f'y_{i + 1}e = {error:>6.3f}'
                                           for i, error
                                           in enumerate(y_error)])
            # Print time step, MPC cost value, and formatted error
            print(f"    Time step: {t:>4} - MPC cost value: "
                  f"{mpc_cost_val:>8.4f} - Error: {formatted_u_error}, "
                  f"{formatted_y_error}")
    
    return u_sys, y_sys