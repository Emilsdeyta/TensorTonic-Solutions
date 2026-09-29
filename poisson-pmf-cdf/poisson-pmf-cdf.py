import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    
    pmf = ((math.exp(-lam))*(lam**k))/math.factorial(k)
    cdf = sum((math.exp(-lam)*(lam**i))/math.factorial(i) for i in range(k+1))

    results = {"pmf":pmf, "cdf":cdf}
    return results
           