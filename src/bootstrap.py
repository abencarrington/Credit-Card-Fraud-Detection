import os
from pathlib import Path

def set_project_root(root_name: str = "Credit-Card-Fraud-Detection") -> None:
    """
    Walks up from this file’s location until it finds a folder named `root_name`,
    then chdir’s there. Raises error if not found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == root_name:
            os.chdir(parent)
            return
    raise RuntimeError(f"Could not find project root folder '{root_name}' in {here.parents}")