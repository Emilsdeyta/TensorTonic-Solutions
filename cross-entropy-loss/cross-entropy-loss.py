import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    row_indices = np.arange(len(y_true))
    formula = y_pred[row_indices, y_true]
    return -np.mean(np.log(formula))