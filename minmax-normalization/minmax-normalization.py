import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """

    min = np.min(X, axis=axis, keepdims=True)
    max = np.max(X, axis=axis, keepdims=True)

    data_range = max-min
    safe_range = np.where(data_range > eps, data_range, 1.0)

    return (X-min)/(safe_range)
    