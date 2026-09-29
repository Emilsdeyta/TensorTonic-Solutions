import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    
    x = np.asarray(x)
    pmf = np.where(x==0, 1-p, p)
    mean = p
    variance = p*(1-p)

    results = {"pmf":pmf, "mean":float(mean), "variance":float(variance)}
    return results