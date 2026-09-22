import numpy as np

def angle_between_3d(v: list, w: list) -> float:
    """
    Returns the angle as a float.
    """

    
    scalar = np.dot(v,w)

    mod_v = 0
    mod_w = 0
    
    for i in v:
        mod_v += i**2
        
    for j in w:
        mod_w += j**2

    v1 = np.sqrt(mod_v)
    w1 = np.sqrt(mod_w)
    c = scalar/(v1*w1)

    return np.arccos(c)
        