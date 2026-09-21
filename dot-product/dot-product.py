import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    i = np.asarray(x)
    j = np.asarray(y)

    total = 0
    total += i*j
    
    return float(sum(total))
            