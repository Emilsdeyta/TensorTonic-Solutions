import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    
    a = np.asarray(a)
    b = np.asarray(b)

    dot = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a==0 or norm_b==0:
        return 0.0
        
    result = dot/(norm_a*norm_b)

    return float(result)