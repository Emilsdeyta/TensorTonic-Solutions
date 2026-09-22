from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    mean = float(np.mean(x))
    median = float(np.median(x))

    labels, first_idx, counts = np.unique(x, 
                                          return_index=True,   
                                          return_counts=True)

    max_count = np.max(counts)
    candidate_idx = np.where(counts == max_count)[0]
    
    best_idx = candidate_idx[np.argmin(first_idx[candidate_idx])]
    mode = float(x[best_idx])
    return {"mean":mean, "median":median, "mode":mode}