import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """

    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        m = np.max(x)
        exp_values = np.exp(x-m)
        return exp_values/np.sum(exp_values)

    else:
        m = np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(x-m)
        return exp_values/np.sum(exp_values, axis=1, keepdims=True)