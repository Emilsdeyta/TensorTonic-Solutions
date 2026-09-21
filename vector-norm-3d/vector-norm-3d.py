import numpy as np

def vector_norm_3d(v: list) -> float | np.ndarray:
    """
    Returns a float or a NumPy array.
    """
    
    if isinstance(v[0], list):
        total = []
        for i in v:
            row = 0
            for j in i:
                row += j**2
                result = row**0.5
            total.append(result)
        return np.array(total)
    else:
        row = 0
        for i in v:
            row += i**2
        return float(row**0.5)        
            