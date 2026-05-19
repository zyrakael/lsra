"""
Random seed utilities.
"""

import os
import random

import torch

try:
    import numpy as np
except ImportError:  
    np = None  


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set the global random seed across Python, NumPy and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

