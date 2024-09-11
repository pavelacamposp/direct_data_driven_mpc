from typing import Any

import os
import yaml

def load_yaml_config_params(config_file: str, key: str) -> Any:
    """
    Load and retrieve parameters from a YAML configuration file for a
    specified key.
    
    Args:
        config_file (str): The path to a YAML configuration file.
        key (str): The key corresponding to the parameters to be retrieved
            from the configuration file.
    
    Returns:
        Any: The parameters from the configuration file associated with the
            specified key.
    
    Raises:
        FileNotFoundError: If the YAML configuration file is not found.
        ValueError: If the `key` value is missing in the configuration file.
    """
    # Check if the file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not "
                                "found.")
    
    # Load YAML configuration file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Check if 'key' exists in the config
    if key not in config:
        raise ValueError(f"Missing `{key}` value in the configuration file.")
    
    return config[key]
