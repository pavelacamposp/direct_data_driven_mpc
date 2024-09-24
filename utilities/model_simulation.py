import numpy as np

from utilities.initial_state_estimation import (
    observability_matrix, toeplitz_input_output_matrix,
    estimate_initial_state, calculate_equilibrium_output_from_input,
    calculate_equilibrium_input_from_output)

class LTIModel:
    """
    A class representing a Linear Time-Invariant (LTI) system in state-space
    form.

    The system is defined by its state-space matrices `A`, `B`, `C`, `D`, and 
    can simulate its behavior and perform tasks such as estimating initial
    states and calculating equilibrium points.

    Attributes:
        A (np.ndarray): The State matrix of the system.
        B (np.ndarray): The Input matrix of the system.
        C (np.ndarray): The Output matrix of the system.
        D (np.ndarray): The Feedforward matrix of the system.
        eps_max (float): The upper bound of the system measurement noise.
        n (int): The order of the system (number of states).
        m (int): The number of inputs to the system.
        p (int): The number of outputs of the system.
        x (np.ndarray): The internal state vector of the system.
        Ot (np.ndarray): The observability matrix of the system.
        Tt (np.ndarray): The Toeplitz input-output matrix of the system.
    """
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        eps_max: float = 0.0):
        """
        Initialize a LTI system with state-space matrices `A`, `B`, `C`, `D`.

        Args:
            A (np.ndarray): The State matrix of the system.
            B (np.ndarray): The Input matrix of the system.
            C (np.ndarray): The Output matrix of the system.
            D (np.ndarray): The Feedforward matrix of the system.
            eps_max (float): The upper bound of the system measurement noise.
                Defaults to 0.0.
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.eps_max = eps_max
        # System order and number of inputs and outputs
        self.n = A.shape[0] # System order
        self.m = B.shape[1] # Number of inputs
        self.p = C.shape[0] # Number of outputs

        # System state
        self.x = np.zeros(self.n)

        # Define Toeplitz input-output and observability matrices
        # for initial state estimation
        self.Ot = observability_matrix(A=self.A, C=self.C)
        self.Tt = toeplitz_input_output_matrix(
            A=self.A, B=self.B, C=self.C, D=self.D, t=self.n)

    
    def simulate_step(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Simulate a single time step of the LTI system with a given input `u`
        and measurement noise `w`.

        The system simulation follows the state-space equations:
            x(k+1) = A * x(k) + B * u(k)
            y(k) = C * x(k) + D * u(k) + w(k)

        Args:
            u (np.ndarray): The input vector of shape `(m,)` at the current
                time step, where `m` is the number of inputs.
            w (np.ndarray): The measurement noise vector of shape `(p,)` at
                the current time step, where `p` is the number of outputs.

        Returns:
            np.ndarray: The output vector `y` of shape `(p,)` at the current
                time step, where `p` is the number of outputs.

        Note:
            This method updates the `x` attribute, which represents the
            internal state vector of the system, after simulation.
        """
        # Compute output using the current internal state of the system
        y = self.C @ self.x + self.D @ u + w
        # Update the internal system state
        self.x = self.A @ self.x + self.B @ u
        
        return y

    def simulate(
        self,
        U: np.ndarray,
        W: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Simulate the LTI system over multiple time steps.

        Args:
            U (np.ndarray): An input matrix of shape `(steps, m)` where
                `steps` is the number of time steps and `m` is the number of
                inputs.
            W (np.ndarray): A noise matrix of shape `(steps, p)` where `steps`
                is the number of time steps and `p` is the number of outputs.
            steps (int): The number of simulation steps.

        Returns:
            np.ndarray: The output matrix `Y` of shape `(steps, p)` containing
                the simulated system outputs at each time step.

        Note:
            This method updates the `x` attribute, which represents the
            internal state vector of the system, after each simulation step.
        """
        # Initialize system output
        Y = np.zeros((steps, self.p))

        for k in range(steps):
            Y[k, :] = self.simulate_step(U[k, :], W[k, :])
        
        return Y
    
    def get_initial_state_from_trajectory(
        self,
        U: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the initial state of the system corresponding to an
        input-output trajectory.
         
        This method uses a least squares observer with the Toeplitz
        input-output and observability matrices of the system to estimate its
        initial state from the input (`U`) and output (`Y`) trajectory.

        Args:
            U (np.ndarray): An input vector of shape `(n, )`, where `n` is
                the order of the system.
            Y (np.ndarray): An outputs vector of shape `(n, )`, where `n` is
                the order of the system.
    
        Returns:
            np.ndarray: A vector of shape `(n, )` representing the estimated
                initial state of the system .

        Raises:
            ValueError: If `U` or `Y` are not shaped `(n, )`.
        """
        x_0 = estimate_initial_state(Ot=self.Ot, Tt=self.Tt, U=U, Y=Y)
        
        return x_0
    
    def get_equilibrium_output_from_input(
        self,
        u_eq: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the equilibrium output `y_eq` corresponding to an input
        `u_eq` so they represent an equilibrium pair of the system.

        This method calculates the equilibrium output under zero initial
        conditions using the final value theorem.

        Args:
            u_eq (np.ndarray): A vector of shape `(m, 1)` representing an
                input of the system, where `m` is the number of inputs to the
                system.

        Returns:
            np.ndarray: A vector of shape `(p, 1)` representing the
                equilibrium output `y_eq`, where `p` is the number of outputs
                of the system.
        """
        y_eq = calculate_equilibrium_output_from_input(
            A=self.A, B=self.B, C=self.C, D=self.D, u_eq=u_eq)
        
        return y_eq
    
    def get_equilibrium_input_from_output(
        self,
        y_eq: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the equilibrium input `u_eq` corresponding to an output
        `y_eq` so they represent an equilibrium pair of the system.

        This method calculates the equilibrium input under zero initial
        conditions using the final value theorem.

        Args:
            y_eq (np.ndarray): A vector of shape `(p, 1)` representing an
                output of the system, where `p` is the number of outputs of
                the system.

        Returns:
            np.ndarray: A vector of shape `(m, 1)` representing the
                equilibrium input `u_s`, where `m` is the number of inputs to
                the system.
        """
        u_eq = calculate_equilibrium_input_from_output(
            A=self.A, B=self.B, C=self.C, D=self.D, y_eq=y_eq)
        
        return u_eq
    
    def get_system_order(self) -> int:
        """
        Get the order of the system (number of states).

        Returns:
            int: The system order of the system.
        """
        return self.n
    
    def get_number_inputs(self) -> int:
        """
        Get the number of inputs to the system.

        Returns:
            int: The number of inputs of the system.
        """
        return self.m
    
    def get_number_outputs(self) -> int:
        """
        Get the number of outputs of the system.

        Returns:
            int: The number of outputs of the system.
        """
        return self.p
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the system.

        Returns:
            np.ndarray: The current state vector of the system.
        """
        return self.x
    
    def get_eps_max(self) -> float:
        """
        Get the upper bound of the system measurement noise.

        Returns:
            float: The upper bound of the system measurement noise.
        """
        return self.eps_max
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Set a new state for the system.

        Args:
            state (np.ndarray): A vector of shape `(n, )` representing the 
                new system state, where `n` is the order of the system.

        Raises:
            ValueError: If `state` does not match the dimensions of the
                state vector of the system.
        """
        # Validate state dimensions
        if state.shape != self.x.shape:
            raise ValueError("Incorrect dimensions. Expected state shape "
                             f"{self.x.shape}, but got {state.shape}")
        
        # Update system state
        self.x = state

    def set_eps_max(self, eps_max) -> None:
        """
        Set the upper bound of the system measurement noise.

        Args:
            eps_max (float): The new value for the upper bound of the system
                measurement noise.
        """
        self.eps_max = eps_max