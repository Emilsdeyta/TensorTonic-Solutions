import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    labels, first_idx, counts = np.unique(y_train, 
                                          return_index=True,   
                                          return_counts=True)

    max_count = np.max(counts)
    candidate_idx = np.where(counts == max_count)[0]
    
    best_idx = candidate_idx[np.argmin(first_idx[candidate_idx])]
    majorty = labels[best_idx]

    return np.full(len(X_test), majorty)