import numpy as np

def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Calculate the observability matrix for a state-space system defined by
    state matrix A and output matrix C.

    The observability matrix is constructed over n time-steps, where n is the
    number of states in the system, which corresponds to the dimension of the
    A matrix.

    Args:
        A (np.ndarray): The State matrix of the system.
        C (np.ndarray): The Output matrix of the system.
    
    Returns:
        np.ndarray: The observability matrix of the system.
    """
    # Get number of states
    n = A.shape[0] # Number of states

    Ot = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])
    
    return Ot

def toeplitz_input_output_matrix(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    t: int
) -> np.ndarray:
    """
    Construct a Toeplitz matrix that maps inputs to outputs for a state-space
    system defined by matrices A (state), B (input), C (output) and D
    (feedforward), over the time interval [0, t-1].

    This matrix is used to express the linear response of the system outputs
    to the system inputs, extended over t time steps.

    For t = 3, this matrix takes the form:
    Tt = [[D   0   0],
          [CB  D   0],
          [CAB CB  D]]
    
    Args:
        A (np.ndarray): The State matrix of the system.
        B (np.ndarray): The Input matrix of the system.
        C (np.ndarray): The Output matrix of the system.
        D (np.ndarray): The Feedforward matrix of the system.
        t (int): The number of time steps for the Toeplitz matrix extension.
    
    Returns:
        np.ndarray: The Toeplitz input-output matrix of the system over t
            steps.
            
    Examples:
        >>> import numpy as np
        >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> B = np.array([[1], [1], [0]])
        >>> C = np.array([[1, 0, 2], [0, 1, 0]])
        >>> D = np.array([[0], [1]])
        >>> t = 3
        >>> print(toeplitz_input_output_matrix(A, B, C, D, t))
        [[ 0.  0.  0.]
         [ 1.  0.  0.]
         [ 1.  0.  0.]
         [ 1.  1.  0.]
         [33.  1.  0.]
         [ 9.  1.  1.]]
    """
    if t <= 0:
        raise ValueError("The number of time steps t must be positive.")

    # Get number of inputs and outputs
    m = B.shape[1] # Number of inputs
    p = C.shape[0] # Number of outputs

    # Precompute powers of A
    A_pows = [np.linalg.matrix_power(A, i) for i in range(t)]

    # Construct Toeplitz input-output matrix
    Tt = np.zeros((p * t, m * t))
    for i in range(t):
        for j in range(t):
            if i == j:
                Tt[i * p: (i + 1) * p,
                   j * m: (j + 1) * m] = D
            elif j < i:
                Tt[i * p: (i + 1) * p,
                   j * m: (j + 1) * m] = C @ A_pows[i - j - 1] @ B
    
    return Tt

def estimate_initial_state(
    Ot: np.ndarray,
    Tt: np.ndarray,
    U: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Estimate the initial state of an observable system based on its
    input-output history using a least squares observer with the Toeplitz
    input-output and observability matrices of the system.

    Args:
        Ot (np.ndarray): The observability matrix of the system.
        Tt (np.ndarray): The Toeplitz input-output matrix of the system over
            `t` steps.
        U (np.ndarray): The vector of inputs over the past `t` time steps.
        Y (np.ndarray): The vector of outputs over the past `t` time steps.
    
    Returns:
        np.ndarray: The estimated initial state.
    
    Raises:
        ValueError: If there is a dimension mismatch between the inputs.
    """
    # Check correct matrix dimensions
    if Ot.shape[0] != Y.shape[0]:
        raise ValueError(f"Dimension mismatch: Ot has {Ot.shape[0]} rows but "
                         f"Y has {Y.shape[0]} rows.")
    if Tt.shape[0] != Y.shape[0]:
        raise ValueError(f"Dimension mismatch: Tt has {Tt.shape[0]} rows but "
                         f"Y has {Y.shape[0]} rows.")
    if Tt.shape[1] != U.shape[0]:
        raise ValueError(f"Dimension mismatch: Tt has {Tt.shape[1]} columns "
                         f"but U has {U.shape[0]} rows.")
    
    # Estimate initial state based on input-output data
    initial_x = np.linalg.pinv(Ot) @ (Y - Tt @ U)

    return initial_x

def calculate_output_equilibrium_setpoint(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    u_s: np.ndarray
) -> np.ndarray:
    """
    Calculate the output setpoint `y_s` corresponding to the input setpoint
    `u_s` so they represent an equilibrium pair of the system defined by
    matrices A (state), B (input), C (output) and D (feedforward).

    Args:
        A (np.ndarray): The State matrix of the system.
        B (np.ndarray): The Input matrix of the system.
        C (np.ndarray): The Output matrix of the system.
        D (np.ndarray): The Feedforward matrix of the system.
        u_s (np.ndarray): An input setpoint array of the system.

    Returns:
        np.ndarray: The output equilibrium setpoint `y_s` corresponding to the
            input setpoint `u_s`.
    """
    n = A.shape[0] # Order of the system

    # Calculate equilibrium output using the final value theorem,
    # assuming zero initial conditions
    M = C @ np.linalg.inv(np.eye(n) - A) @ B + D
    y_s = M @ u_s

    return y_s

def calculate_input_equilibrium_setpoint(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    y_s: np.ndarray
) -> np.ndarray:
    """
    Calculate the input setpoint `u_s` corresponding to the output setpoint
    `y_s` so they represent an equilibrium pair of the system defined by
    matrices A (state), B (input), C (output) and D (feedforward).

    Args:
        A (np.ndarray): The State matrix of the system.
        B (np.ndarray): The Input matrix of the system.
        C (np.ndarray): The Output matrix of the system.
        D (np.ndarray): The Feedforward matrix of the system.
        y_s (np.ndarray): An output setpoint array of the system.

    Returns:
        np.ndarray: The input equilibrium setpoint `u_s` corresponding to the
            output setpoint `y_s`.
    """
    n = A.shape[0] # Order of the system

    # Calculate equilibrium input using the final value theorem,
    # assuming zero initial conditions
    M = C @ np.linalg.inv(np.eye(n) - A) @ B + D
    u_s = np.linalg.pinv(M) @ y_s

    return u_s