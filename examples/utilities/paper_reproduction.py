from typing import List, Tuple, Optional
from enum import Enum

import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt

from examples.utilities.controller_creation import (
    DataDrivenMPCParamsDictType, create_data_driven_mpc_controller)
from examples.utilities.controller_operation import (
    simulate_data_driven_mpc_control_loop)
from utilities.data_visualization import (
    plot_input_output, create_figure_subplots)

from utilities.model_simulation import LTIModel
from direct_data_driven_mpc.direct_data_driven_mpc_controller import (
    DirectDataDrivenMPCController)

# Define Data-Driven MPC controller schemes
class DataDrivenMPCScheme(Enum):
    """
    Robust Data-Driven MPC schemes, as presented in the paper example from
    [1].

    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """
    # 1-step Data-Driven MPC scheme with terminal equality constraints
    TEC = 0
    # n-step Data-Driven MPC scheme with terminal equality constraints
    TEC_N_STEP = 1
    # 1-step Data-Driven MPC scheme without terminal equality constraints
    UCON = 2

# Define Data-Driven MPC scheme configurations
DD_MPC_SCHEME_CONFIG = {
    DataDrivenMPCScheme.TEC: {
        'label': 'TEC',
        'n_mpc_step': 1,
        'terminal_constraint': True,
    },
    DataDrivenMPCScheme.TEC_N_STEP: {
        'label': 'TEC, n-step',
        'n_mpc_step': -1, # -1 used as a placeholder for 'n' steps
        'terminal_constraint': True,
    },
    DataDrivenMPCScheme.UCON: {
        'label': 'UCON',
        'n_mpc_step': 1,
        'terminal_constraint': False,
    }
}

# Define Matplotlib line parameters for Data-Driven MPC schemes
DD_MPC_SCHEME_LINE_PARAMS = {
    DataDrivenMPCScheme.TEC: {
        'color': 'blue',
        'linestyle': 'solid',
        'linewidth': 2
    },
    DataDrivenMPCScheme.TEC_N_STEP: {
        'color': 'lime',
        'linestyle': (0, (5, 5)),
        'linewidth': 2
    },
    DataDrivenMPCScheme.UCON: {
        'color': 'black',
        'linestyle': ':',
        'linewidth': 2
    }
}

SETPOINT_LINE_PARAMS = {
    'color': 'red',
    'linestyle': 'solid',
    'linewidth': 2
}

def get_equilibrium_state_from_output(
    system_model: LTIModel,
    y_eq: np.ndarray
) -> np.ndarray:
    """
    Estimate the equilibrium state of a system corresponding to a specified
    system output.

    This function calculates the control input `u_eq` corresponding to the
    given system output `y_eq`. It then repeats both `u_eq` and `y_eq` over
    `n` (the system order) time steps to construct an input-output trajectory,
    which is used to estimate the initial state.

    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        y_eq (np.ndarray): A vector of shape `(p, 1)` representing an output
            of the system, where `p` is the number of system outputs.

    Returns:
        np.ndarray: The estimated initial system state corresponding to the
            equilibrium output.
    """
    # Get system order
    n = system_model.get_system_order()

    # Calculate the control input that corresponds to the equilibrium output
    u_eq = system_model.get_equilibrium_input_from_output(y_eq=y_eq)

    # Construct equilibrium input-output trajectory for a minimal realization
    U_eq = np.tile(u_eq, n)
    Y_eq = np.tile(y_eq, n)
    
    # Estimate the initial state from the input-output trajectory
    x_eq = system_model.get_initial_state_from_trajectory(U=U_eq, Y=Y_eq)
    
    return x_eq

def create_data_driven_mpc_controllers_reproduction(
    controller_config: DataDrivenMPCParamsDictType,
    u_d: np.ndarray,
    y_d: np.ndarray,
    data_driven_mpc_controller_schemes: List[DataDrivenMPCScheme]
) -> List[DirectDataDrivenMPCController]:
    """
    Create `DirectDataDrivenMPCController` instances for a specified list of
    Data-Driven MPC schemes.
    
    This function uses a base Data-Driven MPC controller configuration and
    initial input-output data from a system. Each controller's configuration
    is modified according to the parameters defined in its corresponding
    scheme configuration.

    Note:
        The created controllers are based on the Robust Data-Driven MPC
        controller schemes presented in the paper example from [1].

    Args:
        controller_config (DataDrivenMPCParamsDictType): A dictionary
            containing Data-Driven MPC controller configuration parameters
            used as a base configuration.
        u_d (np.ndarray): An array of shape `(N, m)` representing a
            persistently exciting input sequence used to generate output data
            from the system. `N` is the trajectory length and `m` is the
            number of control inputs.
        y_d (np.ndarray): An array of shape `(N, p)` representing the system's
            output response to `u_d`. `N` is the trajectory length and `p` is
            the number of system outputs.
        data_driven_mpc_controller_schemes (List[DataDrivenMPCScheme]): A list
            of `DataDrivenMPCScheme` objects, which represent Robust
            Data-Driven MPC schemes based on the paper example from [1].

    Returns:
        List[DirectDataDrivenMPCController]: A list of
            `DirectDataDrivenMPCController` instances, which represent
            Data-Driven MPC controllers based on specified configurations.

    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """
    data_driven_mpc_controllers = []
    
    # Create Data-Driven MPC controllers for each scheme
    for dd_mpc_scheme in data_driven_mpc_controller_schemes:
        # Validate Data-Driven MPC scheme configuration
        if dd_mpc_scheme not in DD_MPC_SCHEME_CONFIG:
            raise ValueError(f"Configuration for scheme {dd_mpc_scheme} "
                             "not found.")
        
        # Created a copy from the base controller configuration
        base_controller_config = controller_config.copy()

        # Get the scheme configuration
        scheme_config = DD_MPC_SCHEME_CONFIG[dd_mpc_scheme]

        # Update controller's configuration based on its scheme configuration
        # n-Step Data-Driven MPC parameter
        if scheme_config['n_mpc_step'] == 1:
            # 1-step Data-Driven MPC control
            base_controller_config['n_mpc_step'] = 1
        else:
            # n-step Data-Driven MPC control
            base_controller_config['n_mpc_step'] = base_controller_config['n']
        
        # Terminal constraint use in Data-Driven MPC formulation
        use_terminal_constraint =  scheme_config['terminal_constraint']

        # Create Data-Driven MPC controller based on scheme config
        dd_mpc_controller = create_data_driven_mpc_controller(
                controller_config=base_controller_config,
                u_d=u_d,
                y_d=y_d,
                use_terminal_constraint=use_terminal_constraint)
        
        # Store controller
        data_driven_mpc_controllers.append(dd_mpc_controller)
    
    return data_driven_mpc_controllers

def simulate_data_driven_mpc_control_loops_reproduction(
    system_model: LTIModel,
    data_driven_mpc_controllers: List[DirectDataDrivenMPCController],
    t_sim : int,
    np_random: Generator,
    verbose: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Simulate multiple Data-Driven MPC control loops applied to a system and
    return the resulting input-output data sequences for each controller.

    This function extends the simulation of a Data-Driven MPC control loop to
    reproduce examples involving multiple controllers. It simulates several
    `DirectDataDrivenMPCController` instances independently on the same system
    model by saving the system's initial internal state and resetting it to
    this state before each simulation.

    Args:
        system_model (LTIModel): An `LTIModel` instance representing a Linear
            Time-Invariant (LTI) system.
        data_driven_mpc_controllers (List[DirectDataDrivenMPCController]): A
            list of `DirectDataDrivenMPCController` instances representing the
            Data-Driven MPC controllers to be simulated.
        t_sim (int): The number of time steps for the simulation.
        np_random (Generator): A Numpy random number generator for generating
            random noise for the system's output.
        verbose (int): The verbosity level.
    
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple containing:
            - A list of arrays, each of shape `(t_sim, m)`, representing the
                optimal control inputs applied to the system for each
                controller, where `m` is the number of control inputs.
            - A list of arrays, each of shape `(t_sim, p)`, representing the
                output response of the system for each controller, where `p`
                is the number of system outputs.
    """
    # Store the internal state of the system to start
    # multiple simulations at the same state
    model_initial_state = system_model.get_state()

    # Initialize simulated input-output data storage
    u_sys_data = []
    y_sys_data = []

    # Simulate Data-Driven MPC controllers
    n_controllers = len(data_driven_mpc_controllers)
    for i, controller in enumerate(data_driven_mpc_controllers):
        if verbose:
            print(f"Simulating controller {i + 1}/{n_controllers}")
        
        # Reset system internal state
        system_model.set_state(state=model_initial_state)

        # Simulate controller
        u_sys, y_sys = simulate_data_driven_mpc_control_loop(
            system_model=system_model,
            data_driven_mpc_controller=controller,
            t_sim=t_sim,
            np_random=np_random,
            verbose=verbose)
    
        # Store input-output data
        u_sys_data.append(u_sys)
        y_sys_data.append(y_sys)

    return u_sys_data, y_sys_data

def plot_input_output_reproduction(
    data_driven_mpc_controller_schemes: List[DataDrivenMPCScheme],
    u_data: List[np.ndarray],
    y_data: List[np.ndarray],
    u_s: np.ndarray,
    y_s: np.ndarray,
    u_ylimits: Optional[List[Tuple[float, float]]],
    y_ylimits: Optional[List[Tuple[float, float]]],
    figsize: Tuple[int, int] =(14, 8),
    dpi: int = 300,
    fontsize: int = 12
) -> None:
    """
    Plot input-output data with setpoints from multiple Data-Driven MPC
    controller scheme simulations in a Matplotlib figure.

    This function creates 2 rows of subplots: the first row for control
    inputs, and the second for system outputs. Each subplot shows the
    data series corresponding to each controller scheme alongside their
    setpoints as a constant line. The appearance of plot lines is predefined
    for each controller scheme.

    Args:
        data_driven_mpc_controller_schemes (List[DataDrivenMPCScheme]): A list
            of `DataDrivenMPCScheme` objects representing Robust Data-Driven
            MPC schemes.
        u_data (List[np.ndarray]): A list of arrays containing control input
            data from each controller scheme simulation.
        y_data (List[np.ndarray]): A list of arrays containing system output
            data from each controller scheme simulation.
        u_s (np.ndarray): An array of shape `(m, 1)` containing the `m` input
            setpoint values considered for the controller simulations.
        y_s (np.ndarray): An array of shape `(p, 1)` containing the `p` output
            setpoint values considered for the controller simulations.
        u_ylimits (Optional[List[Tuple[float, float]]]): A list of tuples
            specifying the Y-axis limits for the input subplots.
        y_ylimits (Optional[List[Tuple[float, float]]]): A list of tuples
            specifying the Y-axis limits for the output subplots.
        figsize (Tuple[int, int]): The (width, height) dimensions of the
            created Matplotlib figure.
        dpi (int): The DPI resolution of the figure.
        fontsize (int): The fontsize for labels, legends and axes ticks.
    """
    # Retrieve number of input and output data sequences and their length
    m = u_data[0].shape[1] # Number of inputs
    p = y_data[0].shape[1] # Number of outputs

    # Create example figure subplots
    _, axs_u, axs_y = create_figure_subplots(
        m=m, p=p, figsize=figsize, dpi=dpi, fontsize=fontsize)

    # Plot data iterating through each controller scheme
    for i, dd_mpc_scheme in enumerate(data_driven_mpc_controller_schemes):
        # Get the scheme configuration
        scheme_config = DD_MPC_SCHEME_CONFIG[dd_mpc_scheme]

        # Get Data-Driven MPC scheme line parameters for plotting
        controller_line_params = DD_MPC_SCHEME_LINE_PARAMS[dd_mpc_scheme]

        # Plot input-output data for scheme
        plot_input_output(u_k=u_data[i],
                          y_k=y_data[i],
                          u_s=u_s,
                          y_s=y_s,
                          inputs_line_params=controller_line_params,
                          outputs_line_params=controller_line_params,
                          setpoints_line_params=SETPOINT_LINE_PARAMS,
                          data_label=f" (${scheme_config['label']}$)",
                          u_ylimits=u_ylimits,
                          y_ylimits=y_ylimits,
                          axs_u=axs_u,
                          axs_y=axs_y,
                          dpi=dpi,
                          fontsize=fontsize)
    
    # Show plot
    plt.show()