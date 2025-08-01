"""
Configuration utility for loading and managing project settings.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def get_data_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract data paths from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing data paths
    """
    return config.get('data', {})


def get_feature_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract feature engineering parameters from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing feature parameters
    """
    return config.get('features', {})


def get_model_params(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Extract model parameters from configuration.
    
    Args:
        config: Configuration dictionary
        model_name: Name of the model
        
    Returns:
        Dictionary containing model parameters
    """
    models_config = config.get('models', {})
    return models_config.get(model_name, {})


def get_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training parameters from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing training parameters
    """
    return config.get('training', {}) 