import numpy as np

class FourTankSysParams:
    """
    Parameters of the linearized version of a four tank system considered in
    J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer, "Data-Driven Model
    Predictive Control With Stability and Robustness Guarantees," in IEEE
    Transactions on Automatic Control, vol. 66, no. 4, pp. 1702-1717, April
    2021, doi: 10.1109/TAC.2020.3000182.

    This class encapsulates the system matrices (A, B, C, D) used in the
    linearized state-space representation of the system, and the control
    setpoints (us, ys).

    The linearized model is given by:
        x(k+1) = A * x(k) + B * u(k)
        y(k) = C * x(k) + D * u(k)
    
    The control goal of the system, as defined in the paper, is given by
    (us, ys).

    Attributes:
        A (np.ndarray): System state matrix.
        B (np.ndarray): Input matrix.
        C (np.ndarray): Output matrix.
        D (np.ndarray): Feedforward matrix.
        us (np.ndarray): Setpoint for control input.
        ys (np.ndarray): Setpoint for output.
    """
    A = np.array([[0.921, 0, 0.041, 0],
                  [0, 0.918, 0, 0.033],
                  [0, 0, 0.924, 0],
                  [0, 0, 0, 0.937]], dtype=float)

    B = np.array([[0.017, 0.001],
                  [0.001, 0.023],
                  [0, 0.061],
                  [0.072, 0]], dtype=float)

    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)

    D = np.zeros((2, 2), dtype=float)

    # Control goal parameters
    us = np.array([1, 1], dtype=float)
    ys = np.array([0.65, 0.77], dtype=float)