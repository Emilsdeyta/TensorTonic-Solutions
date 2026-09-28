import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    x = np.asarray(x)
    q = np.asarray(q)

    r = q*(len(x)-1)/100
    x = np.sort(x)
    xl = np.floor(r).astype(int)
    xu = np.ceil(r).astype(int)
    w = r - np.floor(r)
    
    P = (1-w)*x[xl] + w*x[xu]

    return P