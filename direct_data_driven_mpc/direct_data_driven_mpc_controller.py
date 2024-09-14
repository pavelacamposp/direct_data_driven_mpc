from typing import List, Optional
from enum import Enum

import numpy as np
import cvxpy as cp

from direct_data_driven_mpc.utilities.hankel_matrix import (
    hankel_matrix, evaluate_persistent_excitation)

# Define Direct Data-Driven MPC Controller Types
class DataDrivenMPCType(Enum):
    NOMINAL = 0, # Nominal Data-Driven MPC
    ROBUST = 1 # Robust Data-Driven MPC

# Define Slack Variable Constraint Types for Robust Data-Driven MPC
class SlackVarConstraintTypes(Enum):
    NON_CONVEX = 0, # A Non-Convex slack variable constraint
    CONVEX = 1, # A Convex slack variable constraint
    NONE = 2 # Omits an explicit constraint. The slack variable
    # constraint is implicitly satisfied

class DirectDataDrivenMPCController():
    """
    A class that implements a Data-Driven Model Predictive Control (MPC)
    controller. This controller can be configured as either a Nominal or a
    Robust controller. The implementation is based on research by J.
    Berberich et al., as described in [1].

    Attributes:
        controller_type (DataDrivenMPCType): The Data-Driven MPC controller
            type.
        n (int): The estimated order of the system.
        m (int): The number of control inputs.
        p (int): The number of system outputs.
        u_d (np.ndarray): A persistently exciting input sequence.
        y_d (np.ndarray): The system's output response to `u_d`.
        N (int): The initial input (`u_d`) and output (`y_d`) trajectory
            length.
        u_past (np.ndarray): The past `n` input measurements (u[t-n, t-1]).
        y_past (np.ndarray): The past `n` output measurements (y[t-n, t-1]).
        L (int): The prediction horizon length.
        Q (np.ndarray): The output weighting matrix for the MPC formulation.
        R (np.ndarray): The input weighting matrix for the MPC formulation.
        u_s (np.ndarray): The setpoint for control inputs.
        y_s (np.ndarray): The setpoint for system outputs.
        eps_max (Optional[float]): The estimated upper bound of the system
            measurement noise.
        lamb_alpha (Optional[float]): The ridge regularization base weight for
            `alpha`, scaled by `eps_max`.
        lamb_sigma (Optional[float]): The ridge regularization weight for
            `sigma`.
        c (Optional[float]): A constant used to define a Convex constraint for
            the slack variable `sigma` in a Robust MPC formulation.
        slack_var_constraint_type (SlackVarConstraintTypes): The constraint
            type for the slack variable `sigma` in a Robust MPC formulation.
        HLn_ud (np.ndarray): The Hankel matrix constructed from the input data
            `u_d`.
        HLn_yd (np.ndarray): The Hankel matrix constructed from the output
            data `y_d`.
        alpha (cp.Variable): The optimization variable for a data-driven
            input-output trajectory characterization of the system.
        ubar (cp.Variable): The predicted control input variable.
        ybar (cp.Variable): The predicted system output variable.
        sigma (Optional[cp.Variable]): The slack variable to account for noisy
            measurements in a Robust Data-Driven MPC.
        dynamics_constraint (List[cp.Constraint]): The system dynamics
            constraints for a Data-Driven MPC formulation.
        internal_state_constraint (List[cp.Constraint]): The internal state
            constraints for a Data-Driven MPC formulation.
        terminal_constraint (List[cp.Constraint]): The terminal state
            constraints for a Data-Driven MPC formulation.
        slack_var_constraint (List[cp.Constraint]): The slack variable
            constraints for a Robust Data-Driven MPC formulation.
        constraints (List[cp.Constraint]): The combined constraints for the
            Data-Driven MPC formulation.
        cost (cp.Expression): The cost function for the Data-Driven MPC
            formulation.
        problem (cp.Problem): The quadratic programming problem for the
            Data-Driven MPC.
        optimal_u (np.ndarray): The optimal control input derived from the
            Data-Driven MPC solution.

    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """
    def __init__(
        self,
        n: int,
        m: int,
        p: int,
        u_d: np.ndarray,
        y_d: np.ndarray,
        L: int,
        Q: np.ndarray,
        R: np.ndarray,
        u_s: np.ndarray,
        y_s: np.ndarray,
        eps_max: Optional[float] = None,
        lamb_alpha: Optional[float] = None,
        lamb_sigma: Optional[float] = None,
        c: Optional[float] = None,
        slack_var_constraint_type: SlackVarConstraintTypes = (
            SlackVarConstraintTypes.CONVEX),
        controller_type: DataDrivenMPCType = DataDrivenMPCType.NOMINAL,
    ):
        """
        Initialize a Direct Data-Driven MPC with specified system model
        parameters, an initial input-output data trajectory measured from the
        system, Data-Driven MPC parameters, and a specified Data-Driven MPC
        controller type.

        Note:
            The input data `u_d` used to excite the system to get the initial
            output data must be persistently exciting of order (L + 2 * n), as
            defined by the Data-Driven MPC formulations in [1].
        
        Args:
            n (int): The estimated order of the system.
            m (int): The number of control inputs.
            p (int): The number of system outputs.
            u_d (np.ndarray): A persistently exciting input sequence.
            y_d (np.ndarray): The system's output response to `u_d`.
            L (int): The prediction horizon length.
            Q (np.ndarray): The output weighting matrix for the MPC
                formulation.
            R (np.ndarray): The input weighting matrix for the MPC
                formulation.
            u_s (np.ndarray): The setpoint for control inputs.
            y_s (np.ndarray): The setpoint for system outputs.
            eps_max (Optional[float]): The estimated upper bound of the system
                measurement noise.
            lamb_alpha (Optional[float]): The ridge regularization base weight
                for `alpha`. It is scaled by `eps_max`.
            lamb_sigma (Optional[float]): The ridge regularization weight for
                `sigma`.
            c (Optional[float]): A constant used to define a Convex constraint
                for the slack variable `sigma` in a Robust MPC formulation.
            slack_var_constraint_type (SlackVarConstraintTypes): The
                constraint type for the slack variable `sigma` in a Robust MPC
                formulation.
            controller_type (DataDrivenMPCType): The Data-Driven MPC
                controller type.
        """
        # Set controller type
        self.controller_type = controller_type # Nominal or Robust Controller

        # Validate controller type
        controller_types = [DataDrivenMPCType.NOMINAL,
                            DataDrivenMPCType.ROBUST]
        if controller_type not in controller_types:
            raise ValueError("Unsupported controller type.")

        # Define system model
        # System Model
        self.n = n # Estimated system order
        self.m = m # Number of inputs
        self.p = p # Number of outputs

        # Initial Input-Output trajectory data
        self.u_d = u_d # Input trajectory data
        self.y_d = y_d # Output trajectory data
        self.N = u_d.shape[0] # Initial input-output trajectory length

        # Initialize storage variables for past `n` input-output measurements
        # (used for the internal state constraint that ensures predictions
        # align with the internal state of the system's trajectory)
        self.u_past = u_d[-n:,:].reshape(-1, 1) # u[t-n, t-1]
        self.y_past = y_d[-n:,:].reshape(-1, 1) # y[t-n, t-1]

        # Define MPC parameters
        self.L = L # Prediction horizon
        self.Q = Q # Output weighting matrix
        self.R = R # Input weighting matrix

        # Define Input-Output setpoint pair
        self.u_s = u_s # Control input setpoint
        self.y_s = y_s # System output setpoint
        
        # Define Robust MPC parameters
        self.eps_max = eps_max # Upper limit of bounded measurement noise
        self.lamb_alpha = lamb_alpha # Ridge regularization base weight for
        # alpha. It is scaled by `eps_max`.

        self.lamb_sigma = lamb_sigma # Ridge regularization weight for sigma
        # (If large enough, can neglect noise constraint)

        self.c = c # Convex slack variable constraint:
        # ||sigma||_inf <= c * eps_max

        self.slack_var_constraint_type = slack_var_constraint_type # Slack
        # variable constraint type

        # Validate slack variable constraint type
        slack_var_constraint_types = [SlackVarConstraintTypes.NON_CONVEX,
                                      SlackVarConstraintTypes.CONVEX,
                                      SlackVarConstraintTypes.NONE]
        if slack_var_constraint_type not in slack_var_constraint_types:
            raise ValueError("Unsupported slack variable constraint type.")

        # Ensure correct parameter definition for Robust MPC controller
        if self.controller_type == DataDrivenMPCType.ROBUST:
            if None in (eps_max, lamb_alpha, lamb_sigma, c):
                raise ValueError("All robust MPC parameters (eps_max, "
                                 "lamb_alpha, lamb_sigma, c) must be "
                                 "provided for a 'ROBUST' controller.")
            
        # Evaluate if input trajectory data is persistently exciting of
        # order (L + 2 * n)
        self.evaluate_input_persistent_excitation()

        # Check correct prediction horizon length and cost matrix dimensions
        self.check_prediction_horizon_length()
        self.check_weighting_matrices_dimensions()

        # Initialize Data-Driven MPC controller
        self.initialize_data_driven_mpc()
    
    def evaluate_input_persistent_excitation(self) -> None:
        """
        Evaluate whether the input data is persistently exciting of order
        (L + 2 * n).

        This method first verifies that the length of the elements in the
        input data matches the number of inputs of the system. Then, it
        ensures that the length of the input sequence meets a minimum
        threshold, as defined in Remark 1 [1]. Finally, it evaluates the rank
        of the Hankel matrix induced by the input sequence to determine if the
        input sequence is persistently exciting, as described in Definition 1
        [1].
        
        Raises:
            ValueError: If the length of the elements in the data sequence
                does not match the number of inputs of the system, or if the
                input data is not persistently exciting of order (L + 2 * n).
        
        References:
            [1]: See class-level docstring for full reference details.
        """
        # Get the length of the elements of the data sequence
        u_d_n = self.u_d.shape[1] # m - Number of inputs

        # Check if the number of inputs matches the expected
        # number of inputs of the system
        if u_d_n != self.m:
            raise ValueError("The length of the elements of the data "
                             f"sequence ({u_d_n}) should match the number of "
                             f"inputs of the system ({self.m}).")

        # Compute the minimum required length of the input sequence
        # based on Remark 1 from [1]
        N_min = self.m * (self.L + 2 * self.n) + self.L + 2 * self.n - 1

        # Check if the length of the input sequence is sufficient
        if self.N < N_min:
            raise ValueError(
                "Initial input trajectory data is not persistently exciting "
                "of order (L + 2 * n). It does not satisfy the inequality: "
                "N - L - 2 * n + 1 ≥ m * (L + 2 * n). The required minimum N "
                f"is {N_min}, but got {self.N}.")

        # Evaluate if input data is persistently exciting of order (L + 2 * n)
        # based on the rank of its induced Hankel matrix
        expected_order = self.L + 2 * self.n
        in_hankel_rank, in_pers_exc = evaluate_persistent_excitation(
            X=self.u_d, order=expected_order)
        
        if not in_pers_exc:
            raise ValueError(
                "Initial input trajectory data is not persistently exciting "
                "of order (L + 2 * n). The rank of its induced Hankel matrix "
                f"({in_hankel_rank}) does not match the expected rank ("
                f"{u_d_n * expected_order}).")

    def check_prediction_horizon_length(self) -> None:
        """
        Check if the prediction horizon length, `L`, satisfies the MPC system
        design restriction based on the type of the Direct Data-Driven MPC
        controller. For a `Nominal` type, it must be greater than or equal to
        the estimated system order `n`. For a `Robust` controller, it must be
        greater than or equal to two times the estimated system order `n`.

        These restrictions are defined by Assumption 3, for the Nominal MPC
        scheme, and Assumption 4, for the Robust one, as decribed in [1].

        Raises:
            ValueError: If the prediction horizon `L` is less than the
                required threshold based on the controller type.

        References:
            [1]: See class-level docstring for full reference details.
        """
        if self.controller_type == DataDrivenMPCType.NOMINAL:
            if self.L < self.n:
                raise ValueError("The prediction horizon (`L`) must be "
                                 "greater than or equal to the estimated "
                                 "system order `n`.")
        elif self.controller_type == DataDrivenMPCType.ROBUST:
            if self.L < 2 * self.n:
                raise ValueError("The prediction horizon (`L`) must be "
                                 "greater than or equal to two times the "
                                 "estimated system order `n`.")
        
    def check_weighting_matrices_dimensions(self) -> None:
        """
        Check if the dimensions of the output and input weighting matrices, Q
        and R, are correct for an MPC formulation based on their order.

        Raises:
            ValueError: If the dimensions of the Q or R matrices are incorrect.
        """
        expected_output_shape = (self.p * self.L, self.p * self.L)
        expected_input_shape = (self.m * self.L, self.m * self.L)

        if self.Q.shape != expected_output_shape:
            raise ValueError("Output weighting square matrix Q should be"
                             "of order (p * L)")
        if self.R.shape != expected_input_shape:
            raise ValueError("Input weighting square matrix R should be"
                             "of order (m * L)")
        
    def initialize_data_driven_mpc(self) -> None:
        """
        Initialize the Data-Driven MPC controller.

        This method performs the following tasks:
        1. Construct Hankel matrices from the initial input-output trajectory
            data (`u_d`, `y_d`). These matrices are used for the data-driven
            characterization of the unknown system, as defined by the system
            dynamic constraint in the Robust and Nominal Data-Driven MPC
            formulations from [1].
        2. Define the optimization variables for the Data-Driven MPC problem.
        3. Define the constraints for the MPC problem, which include the
            system dynamics, internal state, terminal state, and, for a Robust
            MPC controller, the slack variable constraint.
        4. Define the cost function for the MPC problem.
        5. Formulates the MPC problem as a Quadratic Programming (QP) problem.
        6. Solves the initialized MPC problem to ensure the formulation is
            valid and retrieve the optimal control input for the initial
            setup.

        This initialization process ensures that all necessary components for
        the Data-Driven MPC are correctly defined and that the MPC problem is
        solvable with the provided initial data.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Construct Hankel Matrices from initial input-output trajectory for
        # the data-driven characterization of the unknown system used for the
        # system dynamic constraint defined by Equation 3b (Nominal MPC) and
        # Equation 6a (Robust MPC).
        self.HLn_ud = hankel_matrix(self.u_d, self.L + self.n) # H_{L+n}(u^d)
        self.HLn_yd = hankel_matrix(self.y_d, self.L + self.n) # H_{L+n}(y^d)
        
        # Define the Data-Driven MPC problem
        self.define_optimization_variables()
        self.define_mpc_constraints()
        self.define_cost_function()
        self.define_mpc_problem()

        # Validate the Data-Driven MPC formulation with an initial solution
        self.solve_mpc_problem()
        self.get_optimal_control_input()

    def update_and_solve_data_driven_mpc(self) -> None:
        """
        Update the Data-Driven MPC problem constraints and formulation, solve
        it and store the optimal control input.

        This method updates the MPC constraints definition and problem
        formulation to account for updated past `n` input-output measurements
        of the system. This stored historical data is used for the Internal
        State constraint defined by Equation 3c (Nominal) and Equation 6b
        (Robust) from [1]. Then, the method solves the MPC problem and stores
        the optimal control input derived from the solution.

        References:
            [1]: See class-level docstring for full reference details.
        """
        self.define_mpc_constraints()
        self.define_mpc_problem()
        self.solve_mpc_problem()
        self.get_optimal_control_input()

    def define_optimization_variables(self) -> None:
        """
        Define the optimization variables for the Data-Driven MPC formulation
        based on the specified MPC controller type.

        This method defines data-driven MPC optimization variables as
        described in the Nominal and Robust MPC formulations in [1]:
        - **Nominal MPC**: Defines the variable `alpha` for a data-driven
            input-output trajectory characterization of the system, and the
            predicted input (`ubar`) and output (`ybar`) variables, as
            described in Equation 3.
        - **Robust MPC**: In addition to the optimization variables defined
            for a Nominal MPC formulation, defines the `sigma` variable to
            account for noisy measurements, as described in Equation 6.
        
        Note:
            This method initializes the `alpha`, `ubar`, `ybar`, and `sigma`
            attributes to define the MPC optimization variables based on the
            MPC controller type. The `sigma` variable is only initialized for
            a Robust MPC controller.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # alpha(t)
        self.alpha = cp.Variable((self.N - self.L - self.n + 1, 1))
        # ubar[-n, L-1](t)
        self.ubar = cp.Variable(((self.L + self.n) * self.m, 1))
        # ybar[-n, L-1](t)
        self.ybar = cp.Variable(((self.L + self.n) * self.p, 1))
        # The time indices of the predicted input and output start at k = −n,
        # since the last `n` inputs and outputs are used to invoke a unique
        # initial state at time `t`, as described in Definition 3 from [1].
        
        if self.controller_type == DataDrivenMPCType.ROBUST:
            # sigma(t)
            self.sigma = cp.Variable(((self.L + self.n) * self.p, 1))
    
    def define_mpc_constraints(self) -> None:
        """
        Define the constraints for the Data-Driven MPC formulation based on
        the specified MPC controller type.

        This method defines the following constraints, as described in the
        Nominal and Robust MPC formulations in [1]:
        - **System dynamics**: Ensures input-output predictions are possible
            trajectories of the system based on a data-driven characterization
            of all its input-output trajectories. In a Robust MPC scheme, adds
            a slack variable to account for noisy measurements. Defined by
            Equation 3b (Nominal) and Equation 6a (Robust).
        - **Internal state**: Ensures predictions align with the internal
            state of the system's trajectory. This constrains the first `n`
            input-output predictions to match the past `n` input-output
            measurements of the system, guaranteeing that the predictions
            consider the initial state of the system. Defined by Equation 3c
            (Nominal) and Equation 6b (Robust).
        - **Terminal state**: Aims to stabilize the internal state of the
            system so it aligns with the steady-state that corresponds to the
            input-output pair (`u_s`, `y_s`) in any minimal realization (last
            `n` input-output predictions, as considered in [1]). Defined by
            Equation 3d (Nominal) and Equation 6c (Robust).
        - **Slack Variable**: Bounds a slack variable that accounts
            for noisy online measurements and for noisy data used for
            prediction (used to construct the Hankel matrices). Defined by
            Equation 6d, for a Non-Convex constraint, and Remark 3, for a
            Convex constraint and an implicit alternative.
        
        Note:
            This method initializes the `dynamics_constraint`,
            `internal_state_constraint`, `terminal_constraint`,
            `slack_var_constraint`, and `constraints` attributes to define the
            MPC constraints based on the MPC controller type.
        
        References:
            [1]: See class-level docstring for full reference details.
        """
        # Define System Dynamic, Internal State and Terminal State Constraints
        self.dynamics_constraint = self.define_system_dynamic_constraint()
        self.internal_state_constraint = (
            self.define_internal_state_constraint())
        self.terminal_constraint = self.define_terminal_state_constraint(
            u_s=self.u_s, y_s=self.y_s)
        
        # Define Slack Variable Constraint if controller type is Robust
        self.slack_var_constraint = (
            self.define_slack_variable_constraint()
            if self.controller_type == DataDrivenMPCType.ROBUST
            else [])
            
        # Combine constraints
        self.constraints = (self.dynamics_constraint +
                            self.internal_state_constraint +
                            self.terminal_constraint +
                            self.slack_var_constraint)

    def define_system_dynamic_constraint(self) -> List[cp.Constraint]:
        """
        Define the system dynamic constraint for the Data-Driven MPC
        formulation corresponding to the specified MPC controller type.

        This constraint uses a data-driven characterization of all the
        input-output trajectories of a system, as defined by Theorem 1 [1], to
        ensure predictions are possible system trajectories. This is analogous
        to the system dynamics contraint in a typical MPC formulation.

        In a Robust MPC scheme, this constraint adds a slack variable to
        account for noisy online measurements and for noisy data used for
        prediction (used to construct the Hankel matrices).

        The constraint is defined according to the following equations from
        the Nominal and Robust MPC formulations in [1]:
        - Nominal MPC: Equation 3b.
        - Robust MPC: Equation 6a.

        Returns:
            List[cp.Constraint]: A list containing the CVXPY system dynamic
                constraint for the Data-Driven MPC controller, corresponding
                to the specified MPC controller type.
        
        References:
            [1]: See class-level docstring for full reference details.
        """
        if self.controller_type == DataDrivenMPCType.NOMINAL:
            # Define system dynamic constraint for Nominal MPC
            # based on Equation 3b from [1]
            dynamics_constraint = [
                cp.vstack([self.ubar, self.ybar]) ==
                cp.vstack([self.HLn_ud, self.HLn_yd]) @ self.alpha]
        elif self.controller_type == DataDrivenMPCType.ROBUST:
            # Define system dynamic constraint for Robust MPC
            # including a slack variable to account for noise,
            # based on Equation 6a from [1]
            dynamics_constraint = [
                cp.vstack([self.ubar, self.ybar + self.sigma]) ==
                cp.vstack([self.HLn_ud, self.HLn_yd]) @ self.alpha]
        
        return dynamics_constraint
        
    def define_internal_state_constraint(self) -> List[cp.Constraint]:
        """
        Define the internal state constraint for the Data-Driven MPC
        formulation.

        This constraint ensures predictions align with the internal state of
        the system's trajectory. This way, the first `n` input-output
        predictions are constrained to match the past `n` input-output
        measurements of the system, guaranteeing that the predictions consider
        the initial state of the system.

        The constraint is defined according to Equation 3c (Nominal MPC) and
        Equation 6b (Robust MPC) from the Nominal and Robust MPC formulations
        in [1].

        Returns:
            List[cp.Constraint]: A list containing the CVXPY internal state
                constraint for the Data-Driven MPC controller.
        
        Note:
            It is essential to update the past `n` input-output measurements
            of the system, `u_past` and `y_past`, at each MPC iteration.
        
        References:
            [1]: See class-level docstring for full reference details.
        """
        # Define internal state constraint for Nominal and Robust MPC
        # based on Equation 3c and Equation 6b from [1], respectively
        ubar_state = self.ubar[:self.n * self.m] # ubar[-n, -1]
        ybar_state = self.ybar[:self.n * self.p] # ybar[-n, -1]
        internal_state_constraint = [
            cp.vstack([ubar_state, ybar_state]) ==
            cp.vstack([self.u_past, self.y_past])]
        
        return internal_state_constraint
    
    def define_terminal_state_constraint(
        self,
        u_s: np.ndarray,
        y_s: np.ndarray
    ) -> List[cp.Constraint]:
        """
        Define the terminal state constraint for the Data-Driven MPC
        formulation.

        This constraint aims to stabilize the internal state of the system so
        it aligns with the steady-state that corresponds to the input-output
        pair (`u_s`, `y_s`) in any minimal realization, specifically the last
        `n` input-output predictions, as considered in [1].

        The constraint is defined according to Equation 3d (Nominal MPC) and
        Equation 6c (Robust MPC) from the Nominal and Robust MPC formulations
        in [1].

        Returns:
            List[cp.Constraint]: A list containing the CVXPY terminal state
                constraint for the Data-Driven MPC controller.
        
        References:
            [1]: See class-level docstring for full reference details.
        """
        # Get terminal segments of input-output predictions
        # ubar[L-n, L-1]
        ubar_terminal = self.ubar[self.L * self.m :
                                  (self.L + self.n) * self.m]
        # ybar[L-n, L-1]
        ybar_terminal = self.ybar[self.L * self.p :
                                  (self.L + self.n) * self.p]
        
        # Replicate steady-state vectors to match minimum realization
        # dimensions for constraint comparison
        u_sn = np.tile(u_s, (self.n, 1))
        y_sn = np.tile(y_s, (self.n, 1))
        
        # Define terminal state constraint for Nominal and Robust MPC
        # based on Equation 3d and Equation 6c from [1], respectively.
        terminal_constraint = [
            cp.vstack([ubar_terminal, ybar_terminal]) ==
            cp.vstack([u_sn, y_sn])]
        
        return terminal_constraint

    def define_slack_variable_constraint(self) -> List[cp.Constraint]:
        """
        Define the slack variable constraint for a Robust Data-Driven MPC
        formulation based on the specified slack variable constraint type.

        This constraint bounds a slack variable (`sigma`) that accounts for
        noisy online measurements and for noisy data used for prediction (used
        to construct the Hankel matrices for the system dynamic constraint).

        As described in [1], this constraint can be defined in three different
        ways, achieving the same theoretical guarantees:
        - **Non-Convex**: Defines a non-convex constraint (Equation 6d).
        - **Convex**: Defines a convex constraint using a sufficiently large
            coefficient `c` (Remark 3).
        - **None**: Omits an explicit constraint definition. The slack variable
            constraint is implicitly met, relying on a high `lamb_sigma` value
            (Remark 3).

        Returns:
            List[cp.Constraint]: A list containing the CVXPY slack variable
                constraint for the Robust Data-Driven MPC controller,
                corresponding to the specified slack variable constraint type.
                The list is empty if the `NONE` constraint type is selected.
        
        References:
            [1]: See class-level docstring for full reference details.
        """
        # Get prediction segments of sigma variable
        sigma_pred = self.sigma[self.n * self.p:] # sigma[0,L-1]

        # Define slack variable constraint for Robust MPC based
        # on the noise constraint type
        slack_variable_constraint = []
        if (self.slack_var_constraint_type ==
            SlackVarConstraintTypes.NON_CONVEX):
            # Define slack variable constraint considering
            # a NON-CONVEX constraint based on Equation 6d [1]
            slack_variable_constraint = [
                cp.norm(sigma_pred, "inf") <=
                self.eps_max * (1 + cp.norm(self.alpha, 1))]
        elif self.slack_var_constraint_type == SlackVarConstraintTypes.CONVEX:
            # Define slack variable constraint considering
            # a CONVEX constraint based on Remark 3 [1]
            slack_variable_constraint = [cp.norm(sigma_pred, "inf") <=
                                         self.c * self.eps_max]
                
        return slack_variable_constraint

    def define_cost_function(self) -> None:
        """
        Define the cost function for the Data-Driven MPC formulation based on
        the specified MPC controller type.

        This method defines the MPC cost function as described in the Nominal
        and Robust MPC formulations in [1]:
        - **Nominal MPC**: Implements a quadratic stage cost that penalizes
            deviations of the predicted control inputs (`ubar`) and outputs
            (`ybar`) from the desired equilibrium (`u_s`, `y_s`), as described
            in Equation 3.
        - **Robust MPC**: In addition to the quadratic stage cost, adds ridge
            regularization terms for `alpha` and `sigma` variables to account
            for noisy measurements, as described in Equation 6.
        
        Note:
            This method initializes the `cost` attribute to define the MPC
            cost function based on the MPC controller type.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Get segments of input-output predictions from time step 0 to (L - 1)
        # ubar[0,L-1]
        ubar_pred = self.ubar[self.n * self.m: (self.L + self.n) * self.m]
        # ybar[0,L-1]
        ybar_pred = self.ybar[self.n * self.p: (self.L + self.n) * self.p]

        # Define control-related cost
        control_cost = (
            cp.quad_form(ubar_pred - np.tile(self.u_s, (self.L, 1)), self.R) +
            cp.quad_form(ybar_pred - np.tile(self.y_s, (self.L, 1)), self.Q))
        
        # Define noise-related cost if controller type is Robust
        if self.controller_type == DataDrivenMPCType.ROBUST:
            noise_cost = (
                self.lamb_alpha * self.eps_max * cp.norm(self.alpha, 2) ** 2 +
                self.lamb_sigma * cp.norm(self.sigma, 2) ** 2)
        
            # Define MPC cost function for a Robust MPC controller
            self.cost = control_cost + noise_cost
        else:
            # Define MPC cost function for a Nominal MPC controller
            self.cost = control_cost
    
    def define_mpc_problem(self) -> None:
        """
        Define the optimization problem for the Data-Driven MPC formulation.

        Note:
            This method initializes the `problem` attribute to define the MPC
            problem of the Data-Driven MPC controller, which is formulated as
            a Quadratic Programming (QP) problem. It assumes that the `cost`
            (objective function) and `constraints` attributes have already
            been defined.
        """
        # Define QP problem
        objective = cp.Minimize(self.cost)
        self.problem = cp.Problem(objective, self.constraints)
        
    def solve_mpc_problem(self) -> str:
        """
        Solve the optimization problem for the Data-Driven MPC formulation.

        Returns:
            str: The status of the optimization problem after attempting to
                solve it (e.g., "optimal", "optimal_inaccurate", "infeasible",
                "unbounded").

        Note:
            This method assumes that the MPC problem has already been defined.
            It solves the problem and updates the `problem` attribute with the
            solution status.
        """
        self.problem.solve()
        
        return self.problem.status
    
    def get_problem_solve_status(self) -> str:
        """
        Get the solve status of the optimization problem of the Data-Driven MPC
        formulation.

        Returns:
            str: The status of the optimization problem after attempting to
                solve it (e.g., "optimal", "optimal_inaccurate", "infeasible",
                "unbounded").
        """
        return self.problem.status

    def get_optimal_cost_value(self) -> float:
        """
        Get the cost value corresponding to the solved optimization problem of
        the Data-Driven MPC formulation.

        Returns:
            float: The optimal cost value of the solved MPC optimization
                problem.
        """
        return self.problem.value
    
    def get_optimal_control_input(self) -> np.ndarray:
        """
        Retrieve and store the optimal control input from the MPC solution.

        Returns:
            np.ndarray: The predicted optimal control input from time step 0
                to (L - 1).
        
        Raises:
            ValueError: If the MPC problem solution status was not "optimal"
                or "optimal_inaccurate".

        Note:
            This method should be called after the MPC problem has been
            solved. It stores the predicted optimal control input in the
            `optimal_u` attribute.
        """
        # Get segment of the input prediction from time step 0 to (L - 1)
        # ubar[0,L-1]
        ubar_pred = self.ubar[self.n * self.m:
                              (self.L + self.n) * self.m]
        
        # Store the optimal control input ubar*[0,L-1] if the MPC problem
        # solution had an "optimal" or "optimal_inaccurate" status
        if self.problem.status in ["optimal", "optimal_inaccurate"]:
            self.optimal_u = ubar_pred.value.flatten()
            return self.optimal_u
        else:
            raise ValueError("MPC problem was not solved optimally.")
    
    def get_optimal_control_input_at_step(
        self,
        n_step: int = 0
    ) -> np.ndarray:
        """
        Get the optimal control input from the MPC solution corresponding
        to a specified time step in the prediction horizon [0, L-1].

        Args:
            n_step (int): The time step of the optimal control input to
                retrieve. It must be within the range [0, L-1].
        
        Returns:
            np.ndarray: An array containing the optimal control input for the
                specified prediction time step.

        Note:
            This method assumes that the optimal control input from the MPC
            solution has been stored in the `optimal_u` attribute.

        Raises:
            ValueError: If `n_step` is not within the range [0, L-1].
        """
        # Ensure n_step is within prediction range [0,L-1]
        if not 0 <= n_step < self.L:
            raise ValueError(
                f"The specified prediction time step ({n_step}) is out of "
                f"range. It should be within [0, {self.L - 1}].")
        
        optimal_u_step_n = self.optimal_u[n_step * self.m:
                                          (n_step + 1) * self.m]
        
        return optimal_u_step_n
    
    def store_input_output_measurement(
        self,
        u_current: np.ndarray,
        y_current: np.ndarray,
    ) -> None:
        """
        Store an input-output measurement pair for the current time step in
        the input-output storage variables.

        This method updates the input-output storage variables `u_past` and 
        `y_past` by appending the current input-output measurements and
        removing the oldest measurements located at the first position. This
        ensures these variables only store the past `n` measurements, as
        required for the internal state constraint defined by Equation 3c
        (Nominal MPC) and Equation 6b (Robust MPC) [1].

        Args:
            u_current (np.ndarray): The control input for the current
                time step, expected to match the dimensions of prior inputs.
            y_current (np.ndarray): The measured system output for the current
                time step, expected to match the dimensions of prior outputs.
                This output should correspond to the system's response to
                `u_current`, as both represent a trajectory of the system.
    
        Raises:
            ValueError: If `u_current` or `y_current` do not match the
                expected dimensions.
        
        Note:
            This method modifies the `u_past` and `y_past` arrays directly to
            ensure that only the most recent `n` measurements are retained.

        References:
            [1]: See class-level docstring for full reference details.
        """
        # Check measurement dimensions
        expected_u0_dim = (self.m, 1)
        expected_y0_dim = (self.p, 1)
        if (u_current.shape != expected_u0_dim or
            y_current.shape != expected_y0_dim):
            raise ValueError(
                f"Incorrect dimensions. Expected dimensions are "
                f"{expected_u0_dim} for u_current and {expected_y0_dim} for "
                f"y_current, but got {u_current.shape} and "
                f"{y_current.shape} instead.")
        
        # Shift input-output storage arrays to discard the oldest
        # measurements and append the new ones
        # u[t-n, t-1]
        self.u_past = np.vstack([self.u_past[self.m:], u_current])
        # y[t-n, t-1]
        self.y_past = np.vstack([self.y_past[self.p:], y_current])
        
    def set_past_input_output_data(
        self,
        new_u_past: np.ndarray,
        new_y_past: np.ndarray,
    ) -> None:
        """
        Set the storage variables for past input-output measurements.

        This method assigns the provided input-output measurements to the
        arrays storing past input-output measurements, `u_past` and `y_past`.
        It is intended to be used for setting the historical data used in
        the MPC problem formulation.

        Args:
            new_u_past (np.ndarray): An array containing past control inputs.
                Expected to have a shape of (n * m, 1), where 'n' is the
                estimated system order and 'm' is the dimension of the input.
            new_y_past (np.ndarray): An array containing past measured system
                outputs. Expected to have a shape of (n * p, 1) where 'n' is
                the estimated system order and 'p' is the dimension of the
                output.
        
        Raises:
            ValueError: If `new_u_past` or `new_y_past` do not have correct
                dimensions.

        Note:
            This method sets the values of the `u_past` and `y_past`
            attributes with the provided new historical data.
        """
        # Validate input types and dimensions
        expected_u_dim = (self.n * self.m, 1)
        expected_y_dim = (self.n * self.p, 1)
        if new_u_past.shape != expected_u_dim:
            raise ValueError(
                f"Incorrect dimensions. new_u_past must be shaped as "
                f"{expected_u_dim}. Got {new_u_past.shape}. instead")
        if new_y_past.shape != expected_y_dim:
            raise ValueError(
                f"Incorrect dimensions. new_y_past must be shaped as "
                f"{expected_y_dim}. Got {new_y_past.shape} instead.")

        # Update past input-output data
        # u[t-n, t-1]
        self.u_past = new_u_past
        # y[t-n, t-1]
        self.y_past = new_y_past
    
    def set_input_output_setpoints(
        self,
        new_u_s: np.ndarray,
        new_y_s: np.ndarray
    ) -> None:
        """
        Set the control and system setpoints of the Data-Driven MPC controller.

        This method updates the control and system setpoints, `u_s` and `y_s`
        to the provided values `new_u_s` and `new_y_s`. Then, it reinitializes
        the controller to redefine the Data-Driven MPC formulation.

        Args:
            new_u_s (np.ndarray): The setpoint for control inputs.
            new_y_s (np.ndarray): The setpoint for system outputs.
        
        Raises:
            ValueError: If `new_u_s` or `new_y_s` do not have the expected
                dimensions.
            
        Note:
            This method sets the values of the `u_s` and `y_s` attributes with
            the provided new setpoints.
        """
        # Validate input types and dimensions
        if new_u_s.shape != self.u_s.shape:
            raise ValueError(f"Incorrect dimensions. u_s must have shape "
                             f"{self.u_s.shape}, got {new_u_s.shape}")
        if new_y_s.shape != self.y_s.shape:  # Replace with actual expected shape
            raise ValueError(f"Incorrect dimensions. y_s must have shape "
                             f"{self.y_s.shape}, got {new_y_s.shape}")
    
        # Update Input-Output setpoint pairs
        self.u_s = new_u_s
        self.y_s = new_y_s

        # Reinitialize Data-Driven MPC controller
        self.initialize_data_driven_mpc()