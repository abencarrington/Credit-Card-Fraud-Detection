"""
Bootstrap module to handle project paths, initialization, and configuration.

Usage:
    from src.bootstrap import set_project_root
    set_project_root()  # Sets up paths
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Global variables
PROJECT_ROOT = None
CONFIG = {}


def find_project_root() -> Path:
    """
    Find the project root directory based on key files/directories.
    
    Looks for dodo.py, setup.py, or specific folder structure.
    
    Returns:
        Path to the project root directory
    """
    current_dir = Path(os.getcwd()).absolute()
    
    # Check if already at project root
    if any((current_dir / marker).exists() for marker in ['dodo.py', 'setup.py']):
        return current_dir
    
    # Walk up the directory tree looking for markers
    for parent in current_dir.parents:
        if any((parent / marker).exists() for marker in ['dodo.py', 'setup.py']):
            return parent
    
    # If we can't find a marker, use current directory
    logger.warning("Could not find project root markers. Using current directory.")
    return current_dir


def set_project_root(root_path: Path = None) -> Path:
    """
    Set the project root directory and add it to Python path.
    
    Args:
        root_path: Optional explicit path to project root.
                  If None, will try to detect automatically.
    
    Returns:
        Path to the project root directory
    """
    global PROJECT_ROOT
    
    if root_path is None:
        root_path = find_project_root()
    
    PROJECT_ROOT = root_path
    
    # Add to Python path if not already there
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    
    logger.info(f"Project root set to: {PROJECT_ROOT}")
    return PROJECT_ROOT


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    global PROJECT_ROOT
    
    if PROJECT_ROOT is None:
        return set_project_root()
    
    return PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root.
    
    Args:
        relative_path: Path relative to project root
    
    Returns:
        Absolute path
    """
    root = get_project_root()
    return root / relative_path


def load_config(config_path: str = "config.json") -> dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to config file, relative to project root
    
    Returns:
        Configuration dictionary
    """
    global CONFIG
    
    import json
    
    path = resolve_path(config_path)
    
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}
    
    try:
        with open(path, 'r') as f:
            CONFIG = json.load(f)
        logger.info(f"Loaded configuration from {path}")
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        CONFIG = {}
    
    return CONFIG


def get_config() -> dict:
    """
    Get the current configuration.
    
    Returns:
        Configuration dictionary
    """
    global CONFIG
    
    if not CONFIG:
        return load_config()
    
    return CONFIG