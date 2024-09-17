from typing import TypedDict, Tuple, Optional

import numpy as np
import os

from utilities.yaml_config_loading import load_yaml_config_params

from direct_data_driven_mpc.direct_data_driven_mpc_controller import (
    DirectDataDrivenMPCController, DataDrivenMPCType, SlackVarConstraintTypes)

# Directory paths
dirname = os.path.dirname
examples_directory = dirname(dirname(__file__))
config_directory = os.path.join(examples_directory, 'config')

# Controller config file paths
controller_config_file = 'data_driven_mpc_example_params.yaml'
controller_config_path = os.path.join(config_directory,
                                      controller_config_file)

# Controller config key
controller_key_value = 'data_driven_mpc_params'

# Define Data-Driven MPC controller types
DataDrivenMPCTypesMap = {
    0: DataDrivenMPCType.NOMINAL,
    1: DataDrivenMPCType.ROBUST
}

# Define Slack Variable Constraint Types Mapping
# (based on controller configuration file)
SlackVarConstraintTypesMap = {
    0: SlackVarConstraintTypes.NONE,
    1: SlackVarConstraintTypes.CONVEX,
    2: SlackVarConstraintTypes.NON_CONVEX
}

# Define the dictionary type hint of Data-Driven MPC controller parameters
class DataDrivenMPCParamsDictType(TypedDict, total=False):
    u_range: Tuple[float, float]
    N: int
    n: int
    eps_max: float
    L: int
    Q: np.ndarray
    R: np.ndarray
    lamb_alpha: float
    lamb_sigma: float
    c: float
    slack_var_constraint_type: SlackVarConstraintTypes
    controller_type: DataDrivenMPCType
    n_mpc_step: int
    u_s: np.ndarray
    y_s: np.ndarray

def get_data_driven_mpc_controller_params(
    m: int,
    p: int,
    eps_bar: Optional[float] = None
) -> DataDrivenMPCParamsDictType:
    """
    Load and initialize parameters for a Data-Driven MPC controller from a
    configuration file.
    
    The controller parameters are defined based on the Nominal and Robust
    Data-Driven MPC controller formulations from [1]. The number of control
    inputs (`m`) and system outputs (`p`) are used to construct the output
    (`Q`) and input (`R`) weighting matrices. If `eps_bar` is provided, it
    overrides the estimated upper bound of the system measurement noise from
    the configuration file.

    Args:
        m (int): The number of control inputs.
        p (int): The number of system outputs.
        eps_bar (Optional[float]): The estimated upper bound of the system
            measurement noise. If provided, it overrides the corresponding
            value from the configuration file. Defaults to `None`.
    
    Returns:
        DataDrivenMPCParamsDictType: A dictionary of parameters configured for
            the Data-Driven MPC controller.
    
    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """
    # Load controller parameters from config file
    params = load_yaml_config_params(config_file=controller_config_path,
                                     key=controller_key_value)
    
    # Initialize Data-Driven MPC controller parameter dict
    dd_mpc_params = {}

    # --- Define initial Input-Output data generation parameters ---
    # Persistently exciting input range
    dd_mpc_params['u_range'] = params['u_d_range']
    # Initial input-output trajectory length
    dd_mpc_params['N'] = params['N']

    # --- Define Data-Driven MPC parameters ---
    # Estimated system order
    n = params['n']
    dd_mpc_params['n'] = n
    # Estimated upper bound of the system measurement noise
    eps_max = params['epsilon_bar']
    if eps_bar is not None:
        # Override `eps_max` if passed
        eps_max = eps_bar
    dd_mpc_params['eps_max'] = eps_max
    # Prediction horizon
    L = params['L']
    dd_mpc_params['L'] = L
    # Output weighting matrix
    dd_mpc_params['Q'] = params['Q_scalar'] * np.eye(p * L)
    # Input weighting matrix
    dd_mpc_params['R'] = params['R_scalar'] * np.eye(m * L)

    # Define ridge regularization base weight for alpha, preventing
    # division by zero in noise-free conditions
    lambda_alpha_epsilon_bar = params['lambda_alpha_epsilon_bar']
    if eps_max != 0:
        dd_mpc_params['lamb_alpha'] = lambda_alpha_epsilon_bar / eps_max
    else:
        # Set a high value if eps_max is zero
        dd_mpc_params['lamb_alpha'] = 1000.0
    
    # Ridge regularization weight for sigma
    dd_mpc_params['lamb_sigma'] = params['lambda_sigma']

    # Convex slack variable constraint constant (see Remark 3 from [1])
    dd_mpc_params['c'] = 1.0

    # Slack variable constraint type
    slack_var_constraint_type_config = params['slack_var_constraint_type']
    dd_mpc_params['slack_var_constraint_type'] = (
        SlackVarConstraintTypesMap.get(slack_var_constraint_type_config,
                                       SlackVarConstraintTypes.NONE))
    
    # Controller type
    controller_type_config = params['controller_type']
    dd_mpc_params['controller_type'] = (
        DataDrivenMPCTypesMap.get(controller_type_config,
                                  DataDrivenMPCType.ROBUST))

    # Number of consecutive applications of the optimal input
    # for an n-Step Data-Driven MPC Scheme (multi-step)
    dd_mpc_params['n_mpc_step'] = n
    # Defaults to the estimated system order, as defined
    # in Algorithm 2 from [1]

    # Define Input-Output equilibrium setpoint pair
    u_s = params['u_s']
    y_s = params['y_s']
    # Control input setpoint
    dd_mpc_params['u_s'] = np.array(u_s, dtype=float).reshape(-1, 1)
    # System output setpoint
    dd_mpc_params['y_s'] = np.array(y_s, dtype=float).reshape(-1, 1)
    
    return dd_mpc_params

def create_data_driven_mpc_controller(
    controller_config: DataDrivenMPCParamsDictType,
    u_d: np.ndarray,
    y_d: np.ndarray,
    use_terminal_constraint: bool = True
) -> DirectDataDrivenMPCController:
    """
    Create a `DirectDataDrivenMPCController` instance using a specified
    Data-Driven MPC controller configuration and initial input-output
    trajectory data measured from a system.

    Args:
        controller_config (DataDrivenMPCParamsDictType): A dictionary
            containing Data-Driven MPC controller configuration parameters.
        u_d (np.ndarray): An array of shape `(N, m)` representing a
            persistently exciting input sequence used to generate output data
            from the system. `N` is the trajectory length and `m` is the
            number of control inputs.
        y_d (np.ndarray): An array of shape `(N, p)` representing the system's
            output response to `u_d`. `N` is the trajectory length and `p` is
            the number of system outputs.
        use_terminal_constraint (bool): If True, include terminal equality
            constraints in the Data-Driven MPC formulation. If False, the
            controller will not enforce this constraint. Defaults to True.
    
    Returns:
        DirectDataDrivenMPCController: A `DirectDataDrivenMPCController`
            instance, which represents a Data-Driven MPC controller based on
            the specified configuration.
    """
    # Get model parameters from input-output trajectory data
    m = u_d.shape[1] # Number of inputs
    p = y_d.shape[1] # Number of outputs

    # Retrieve Data-Driven MPC controller parameters
    n = controller_config['n'] # Estimated system order
    L = controller_config['L'] # Prediction horizon
    Q = controller_config['Q'] # Output weighting matrix
    R = controller_config['R'] # Input weighting matrix

    u_s = controller_config['u_s'] # Control input setpoint
    y_s = controller_config['y_s'] # System output setpoint

    # Estimated upper bound of the system measurement noise
    eps_max = controller_config['eps_max']
    # Ridge regularization base weight for `alpha` (scaled by `eps_max`)
    lamb_alpha = controller_config['lamb_alpha']
    # Ridge regularization weight for sigma
    lamb_sigma = controller_config['lamb_sigma']
    # Convex slack variable constraint constant
    c = controller_config['c']

    # Slack variable constraint type
    slack_var_constraint_type = controller_config['slack_var_constraint_type']

    # Data-Driven MPC controller type
    controller_type = controller_config['controller_type']

    # n-Step Data-Driven MPC Scheme parameters
    # Number of consecutive applications of the optimal input
    n_mpc_step = controller_config['n_mpc_step']

    # Create Data-Driven MPC controller
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
        controller_type=controller_type,
        n_mpc_step=n_mpc_step,
        use_terminal_constraint=use_terminal_constraint)
    
    return direct_data_driven_mpc_controller

if __name__ == "__main__":
    print(get_data_driven_mpc_controller_params(1,1))