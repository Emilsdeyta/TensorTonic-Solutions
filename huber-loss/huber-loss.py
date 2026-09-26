import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    e = np.abs(y_true-y_pred)

    first = 0.5*(e**2)
    second = delta*(e-0.5*delta)
    
    result = np.where(delta>=e, first, second)
    return float(np.mean(result))