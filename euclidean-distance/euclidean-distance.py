import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    total = 0
    total += sum((y-x)**2)
    return total**0.5