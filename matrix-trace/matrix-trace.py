import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    total = 0
    A = np.asarray(A)
    for i in range(len(A)):
        total += A[i][i]
    return total
        