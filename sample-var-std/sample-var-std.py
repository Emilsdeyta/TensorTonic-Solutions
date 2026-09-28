import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """

    x = np.asarray(x)
    mean_x = np.mean(x)
    n = len(x)

    s2 = (1/(n-1))*np.sum((x-mean_x)**2)
    s = np.sqrt(s2)

    results = {"variance" : float(s2), "standard_deviation" : float(s)
              }
    return results
    