import torch
import numpy
import random


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed to set.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
