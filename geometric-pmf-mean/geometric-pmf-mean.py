import numpy as np

def geometric_pmf_mean(k: list, p: float) -> dict:
    """
    Returns a dictionary with pmf and mean.
    """
    k = np.asarray(k)
    PMF = ((1-p)**(k-1))*p
    E = 1/p
    results = {"pmf":PMF, "mean":E}
    return results