import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_prediction = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_prediction)**2)
    ss_res = np.sum((y_true-y_pred)**2)
    if ss_res==0 and ss_tot==0:
        return 1.0
    elif ss_tot==0:
        return 0.0

    
    r2 = 1 - (ss_res/ss_tot)

    return float(r2)