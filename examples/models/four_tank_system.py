import numpy as np
import os

from utilities.yaml_config_loading import load_yaml_config_params

from utilities.model_simulation import LTIModel

# Directory paths
dirname = os.path.dirname
models_directory = dirname(__file__)
models_config_directory = os.path.join(models_directory, 'config')

# Model config file paths
model_config_file = 'four_tank_system_params.yaml'
model_config_path = os.path.join(models_config_directory, model_config_file)

# Model config key
model_key_value = 'FourTankSystem'

class FourTankSystem(LTIModel):
    """
    A class that defines the model of a linearized version of a four-tank
    system.

    Attributes:
        A (np.ndarray): System state matrix.
        B (np.ndarray): Input matrix.
        C (np.ndarray): Output matrix.
        D (np.ndarray): Feedforward matrix.
        eps_max (float): Upper bound of the system measurement noise.

    Note:
        The model is based on the linearized version of the four-tank system
        from the example implementation in Section V from [1].
    
    References:
        [1] J. Berberich, J. Köhler, M. A. Müller and F. Allgöwer,
            "Data-Driven Model Predictive Control With Stability and
            Robustness Guarantees," in IEEE Transactions on Automatic Control,
            vol. 66, no. 4, pp. 1702-1717, April 2021,
            doi: 10.1109/TAC.2020.3000182.
    """
    def __init__(self, verbose: int = 0):
        """
        Initialize a Four-Tank system model by loading parameters from a
        predefined YAML configuration file.

        Args:
            verbose (int): The verbose level.
        """
        self.verbose = verbose
        
        # Load model parameters from config file
        params = load_yaml_config_params(config_file=model_config_path,
                                         key=model_key_value)

        if self.verbose:
            print(f"Loaded model parameters from {model_config_path}")
        
        # Get model parameters
        A = np.array(params['A'], dtype=float)
        B = np.array(params['B'], dtype=float)
        C = np.array(params['C'], dtype=float)
        D = np.array(params['D'], dtype=float)
        eps_max = params['eps_max']
        
        # Initialize the base LTIModel class with the loaded parameters
        super().__init__(A=A, B=B, C=C, D=D, eps_max=eps_max)
