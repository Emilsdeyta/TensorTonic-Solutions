import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    if isinstance(x, (int, float)):
        return 1/(1+np.exp(-x))
    elif isinstance(x, (list)) and all(isinstance(i, (int, float)) for i in x):
        return [1/(1+np.exp(-i)) for i in x]
    
    else:
        result = []
        for i in x:
            row = []
            for j in i:
                a = 1/(1+np.exp(-j))
                row.append(a)
            result.append(row)
        return result
            
