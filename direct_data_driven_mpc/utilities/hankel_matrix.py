import numpy as np

def hankel_matrix(X: np.ndarray, L: int) -> np.ndarray:
    """
    Construct a Hankel matrix from the input data matrix `X` with a window
    length `L`. The matrix `X` consists of a sequence of `N` elements, each of
    length `n`.
    
    Args:
        X (np.ndarray): Input data matrix of shape (N, n), where N is the
            number of elements, and n is the length of each element.
        L (int): Window length for the Hankel matrix.
    
    Returns:
        np.ndarray: A Hankel matrix of shape (L * n, N - L + 1), where each 
            column represents a flattened window of length L sliding over the
            N data elements.
    
    Raises:
        ValueError: If the number of elements N is less than the window length
            L, indicating that the window length exceeds the available data
            length.
    
    Examples:
        >>> import numpy as np
        >>> N = 4 # Data length
        >>> L = 2 # Hankel matrix window length
        >>> n = 2 # Data vector length
        >>> rng = np.random.default_rng(0) # RNG for reproducibility
        >>> u_d = rng.uniform(-1, 1, (N, n)) # Generate data matrix
        >>> print(hankel_matrix(u_d, L))
        [[ 0.27392337 -0.91805295  0.62654048]
         [-0.46042657 -0.96694473  0.82551115]
         [-0.91805295  0.62654048  0.21327155]
         [-0.96694473  0.82551115  0.45899312]]
    """
    # Get data matrix shape
    N, n = X.shape
    
    # Validate input dimensions
    if N < L:
        raise ValueError("N must be greater than or equal to L.")
    
    # Initialize the Hankel matrix induced by {x_k}_{k=0}^{N-1}
    HL = np.zeros((L * n, N - L + 1))

    # Construct Hankel matrix
    for i in range(N - L + 1):
        HL[:, i] = X[i: i + L, :].flatten()
    
    return HL
