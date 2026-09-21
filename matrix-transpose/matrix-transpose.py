import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """

    N = len(A)
    M = len(A[0])

    result = []
    for j in range(M):
        row = []
        
        for i in range(N):
            row.append(A[i][j])
        result.append(row)
    return np.array(result)
            
            
